"""Validation evaluation for Milestone 2 recommendation baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .milestone2_data import HISTORY_LENGTHS, Snapshot


METHODS = ("popularity", "item_cosine", "binary_diffusion")
TOP_K = 10


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate and per-user validation results."""

    metrics: pd.DataFrame
    per_user: pd.DataFrame


def evaluate_validation(snapshot: Snapshot) -> EvaluationResult:
    """Evaluate three baselines on every history cohort in one snapshot.

    The target histories are held out of the reference graph. All methods rank
    the same reference-rated movie catalog after masking every movie visible in
    a target user's chosen history. Personalized models use popularity whenever
    they have no usable positive seed or no positive-scoring unseen candidate.
    """
    if snapshot.target_history.empty or snapshot.held_out_positives.empty:
        raise ValueError("The snapshot must contain eligible users and held-out positives.")
    candidates = snapshot.candidate_movies.to_numpy(dtype=np.int64)
    if len(candidates) == 0:
        raise ValueError("The reference graph has no candidate movies.")

    reference_positive = snapshot.positive_edges
    reference_users = pd.Index(
        np.sort(reference_positive["user_id"].unique()), name="user_id"
    )
    movie_to_col = {int(movie_id): col for col, movie_id in enumerate(candidates)}
    user_to_row = {int(user_id): row for row, user_id in enumerate(reference_users)}
    row_indices = reference_positive["user_id"].map(user_to_row).to_numpy(dtype=np.int32)
    col_indices = reference_positive["movie_id"].map(movie_to_col).to_numpy(dtype=np.int32)
    incidence = sparse.csr_matrix(
        (np.ones(len(reference_positive), dtype=np.float64), (row_indices, col_indices)),
        shape=(len(reference_users), len(candidates)),
        dtype=np.float64,
    )
    incidence.sum_duplicates()
    incidence.data[:] = 1.0

    movie_degree = np.asarray(incidence.sum(axis=0)).ravel()
    user_degree = np.asarray(incidence.sum(axis=1)).ravel()
    popularity = movie_degree
    inv_movie_degree = np.divide(
        1.0, movie_degree, out=np.zeros_like(movie_degree), where=movie_degree > 0
    )
    inv_sqrt_movie_degree = np.sqrt(inv_movie_degree)
    inv_user_degree = np.divide(
        1.0, user_degree, out=np.zeros_like(user_degree), where=user_degree > 0
    )

    normalized = incidence.multiply(inv_sqrt_movie_degree).tocsr()
    item_cosine = (normalized.T @ normalized).tocsr()

    target_users = snapshot.target_users.to_numpy(dtype=np.int64)
    target_to_row = {int(user_id): row for row, user_id in enumerate(target_users)}
    held_out_by_user = {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in snapshot.held_out_positives.groupby("user_id", sort=False)
    }
    histories: dict[int, pd.DataFrame] = {
        int(user_id): group
        for user_id, group in snapshot.target_history.groupby("user_id", sort=False)
    }

    metric_rows: list[dict[str, object]] = []
    per_user_rows: list[dict[str, object]] = []
    for history_length in HISTORY_LENGTHS:
        length_name = "full" if history_length is None else str(history_length)
        visible = snapshot.visible_history(history_length)
        visible_by_user = {
            int(user_id): group
            for user_id, group in visible.groupby("user_id", sort=False)
        }
        seed_rows: list[int] = []
        seed_cols: list[int] = []
        seen_by_user: dict[int, set[int]] = {}
        visible_count_by_user: dict[int, int] = {}
        usable_seed_count: dict[int, int] = {}

        for user_id in target_users:
            user_id = int(user_id)
            history = visible_by_user.get(user_id)
            if history is None:
                history = histories[user_id].iloc[0:0]
            visible_count_by_user[user_id] = len(history)
            seen_by_user[user_id] = set(history["movie_id"].astype(int))
            positive_seeds = history.loc[history["rating"].ge(4), "movie_id"].astype(int)
            usable = [movie_id for movie_id in positive_seeds if movie_id in movie_to_col]
            usable_seed_count[user_id] = len(set(usable))
            row = target_to_row[user_id]
            seed_rows.extend([row] * len(set(usable)))
            seed_cols.extend(movie_to_col[movie_id] for movie_id in sorted(set(usable)))

        seeds = sparse.csr_matrix(
            (
                np.ones(len(seed_rows), dtype=np.float64),
                (np.asarray(seed_rows, dtype=np.int32), np.asarray(seed_cols, dtype=np.int32)),
            ),
            shape=(len(target_users), len(candidates)),
            dtype=np.float64,
        )
        personalized_scores = {
            "item_cosine": (seeds @ item_cosine).tocsr(),
            "binary_diffusion": _diffusion_scores(
                seeds, incidence, inv_movie_degree, inv_user_degree
            ),
        }

        per_method: dict[str, list[dict[str, object]]] = {name: [] for name in METHODS}
        for user_id in target_users:
            user_id = int(user_id)
            row = target_to_row[user_id]
            seen = seen_by_user[user_id]
            unseen_mask = np.fromiter(
                (int(movie_id) not in seen for movie_id in candidates),
                dtype=bool,
                count=len(candidates),
            )
            candidate_indices = np.flatnonzero(unseen_mask)
            candidate_ids = candidates[candidate_indices]
            if len(candidate_indices):
                popularity_order = _rank(candidate_ids, popularity[candidate_indices])
                popularity_ids = candidate_ids[popularity_order]
            else:
                popularity_ids = np.asarray([], dtype=np.int64)

            relevant = held_out_by_user[user_id]
            relevant_count = len(relevant)
            for method in METHODS:
                fallback_reason = ""
                if method == "popularity":
                    recommendations = popularity_ids[:TOP_K]
                else:
                    scores = personalized_scores[method].getrow(row).toarray().ravel()
                    candidate_scores = scores[candidate_indices]
                    if usable_seed_count[user_id] == 0:
                        fallback_reason = "no_usable_seed"
                        recommendations = popularity_ids[:TOP_K]
                    elif not np.any(candidate_scores > 0):
                        fallback_reason = "no_positive_unseen_candidate"
                        recommendations = popularity_ids[:TOP_K]
                    else:
                        order = _rank(candidate_ids, candidate_scores)
                        recommendations = candidate_ids[order[:TOP_K]]

                recommendation_list = [int(movie_id) for movie_id in recommendations]
                hit_count = len(set(recommendation_list) & relevant)
                user_result = {
                    "user_id": user_id,
                    "method": method,
                    "history_length": length_name,
                    "visible_ratings": visible_count_by_user[user_id],
                    "top_10_movie_ids": json.dumps(recommendation_list),
                    "hit_count": hit_count,
                    "eligible_relevant_item_count": relevant_count,
                    "fallback_reason": fallback_reason,
                    "precision_at_10": hit_count / TOP_K,
                    "recall_at_10": hit_count / relevant_count,
                }
                per_method[method].append(user_result)
                per_user_rows.append(user_result)

        for method in METHODS:
            rows = per_method[method]
            metric_rows.append({
                "history_length": length_name,
                "method": method,
                "users": len(rows),
                "precision_at_10": float(np.mean([row["precision_at_10"] for row in rows])),
                "recall_at_10": float(np.mean([row["recall_at_10"] for row in rows])),
                "fallback_users": sum(bool(row["fallback_reason"]) for row in rows),
            })

    return EvaluationResult(
        metrics=pd.DataFrame(metric_rows),
        per_user=pd.DataFrame(per_user_rows),
    )


def _diffusion_scores(
    seeds: sparse.csr_matrix,
    incidence: sparse.csr_matrix,
    inv_movie_degree: np.ndarray,
    inv_user_degree: np.ndarray,
) -> sparse.csr_matrix:
    """Apply binary three-step movie-user-movie diffusion to target seeds."""
    weighted_seeds = seeds.multiply(inv_movie_degree).tocsr()
    reached_users = (weighted_seeds @ incidence.T).tocsr()
    reached_users = reached_users.multiply(inv_user_degree).tocsr()
    return (reached_users @ incidence).tocsr()


def _rank(movie_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return deterministic descending-score order with ascending-ID ties."""
    return np.lexsort((movie_ids, -scores))
