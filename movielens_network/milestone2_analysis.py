"""Analyze saved validation recommendations without running any scorers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .milestone2_data import HISTORY_LENGTHS, SPLIT_SEED, VALIDATION_CUTOFF, TEST_CUTOFF, Snapshot
from .milestone2_evaluation import METHODS, TOP_K


HISTORIES = tuple("full" if length is None else str(length) for length in HISTORY_LENGTHS)
PAIRS = (
    ("item_cosine", "popularity"),
    ("binary_diffusion", "popularity"),
    ("binary_diffusion", "item_cosine"),
)
BOOTSTRAP_RESAMPLES = 10_000


def analyze_validation(
    snapshot: Snapshot,
    results_dir: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Validate saved results, summarize paired comparisons, and save tables."""
    source = results_dir / "validation_per_user.csv"
    results = pd.read_csv(source, dtype={"history_length": str}, keep_default_na=False)
    recommendations = _validate_results(results, snapshot)
    summaries = []
    differences = []
    subgroups = []
    popularity_rows = []
    rng = np.random.default_rng(SPLIT_SEED)
    positive_degree = snapshot.positive_edges.groupby("movie_id").size()
    rating_degree = snapshot.reference_ratings.groupby("movie_id").size()
    full_seen = snapshot.target_history.groupby("user_id")["movie_id"].apply(set)

    for length, history_name in zip(HISTORY_LENGTHS, HISTORIES):
        history_results = results.loc[results["history_length"].eq(history_name)]
        visible = snapshot.visible_history(length)
        usable = visible.loc[
            visible["rating"].ge(4) & visible["movie_id"].isin(positive_degree.index)
        ]
        seed_users = set(usable["user_id"])
        cohort = set(snapshot.target_users)
        groups = {
            "all": cohort,
            "usable_seed": seed_users,
            "no_usable_seed": cohort - seed_users,
        }
        for method in METHODS:
            method_results = history_results.loc[history_results["method"].eq(method)]
            summaries.append({
                "history_length": history_name,
                "method": method,
                "users": len(method_results),
                "precision_at_10": method_results["precision_at_10"].mean(),
                "recall_at_10": method_results["recall_at_10"].mean(),
                "fallback_users": method_results["fallback_reason"].ne("").sum(),
            })
            for group_name, user_ids in groups.items():
                selected = method_results.loc[method_results["user_id"].isin(user_ids)]
                subgroups.append({
                    "history_length": history_name,
                    "method": method,
                    "subgroup": group_name,
                    "users": len(selected),
                    "precision_at_10": selected["precision_at_10"].mean(),
                    "recall_at_10": selected["recall_at_10"].mean(),
                    "fallback_users": selected["fallback_reason"].ne("").sum(),
                    "no_usable_seed_fallbacks": selected["fallback_reason"].eq("no_usable_seed").sum(),
                    "no_positive_candidate_fallbacks": selected["fallback_reason"].eq(
                        "no_positive_unseen_candidate"
                    ).sum(),
                })

            recommended_movies = []
            hidden_rated_count = 0
            for row in method_results.itertuples():
                movie_ids = recommendations[(row.user_id, history_name, method)]
                recommended_movies.extend(movie_ids)
                hidden_rated_count += len(set(movie_ids) & full_seen[row.user_id])
            degrees = positive_degree.reindex(recommended_movies, fill_value=0)
            popularity_rows.append({
                "history_length": history_name,
                "method": method,
                "users": len(method_results),
                "recommendation_slots": len(recommended_movies),
                "mean_positive_reference_degree": degrees.mean(),
                "median_positive_reference_degree": degrees.median(),
                "mean_reference_rating_degree": rating_degree.reindex(recommended_movies).mean(),
                "zero_positive_degree_share": degrees.eq(0).mean(),
                "unique_recommended_movies": len(set(recommended_movies)),
                "candidate_movies": len(snapshot.candidate_movies),
                "catalog_coverage": len(set(recommended_movies)) / len(snapshot.candidate_movies),
                "hidden_pre_cutoff_rated_slots": hidden_rated_count,
                "hidden_pre_cutoff_rated_share": hidden_rated_count / len(recommended_movies),
            })

        for metric in ("precision_at_10", "recall_at_10"):
            paired = history_results.pivot(index="user_id", columns="method", values=metric)
            for first, second in PAIRS:
                delta = (paired[first] - paired[second]).to_numpy()
                bootstrapped = delta[rng.integers(len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))]
                low, high = np.quantile(bootstrapped.mean(axis=1), [0.025, 0.975])
                differences.append({
                    "history_length": history_name,
                    "metric": metric,
                    "first_method": first,
                    "second_method": second,
                    "users": len(delta),
                    "mean_difference_pp": delta.mean() * 100,
                    "ci_low_pp": low * 100,
                    "ci_high_pp": high * 100,
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                })

    tables = {
        "validation_history_metrics": pd.DataFrame(summaries),
        "validation_paired_differences": pd.DataFrame(differences),
        "validation_fallback_subgroups": pd.DataFrame(subgroups),
        "validation_recommendation_popularity": pd.DataFrame(popularity_rows),
    }
    saved_metrics = pd.read_csv(
        results_dir / "validation_metrics.csv", dtype={"history_length": str}
    )
    keys = ["history_length", "method"]
    recomputed = tables["validation_history_metrics"].set_index(keys).sort_index()
    saved_metrics = saved_metrics.set_index(keys).sort_index()
    if not recomputed.index.equals(saved_metrics.index) or not np.allclose(
        recomputed.to_numpy(), saved_metrics[recomputed.columns].to_numpy(), atol=1e-9, rtol=0
    ):
        raise ValueError("Saved summary metrics do not match the per-user results.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.10f")
    config = {
        "source_per_user_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_metrics_sha256": hashlib.sha256(
            (results_dir / "validation_metrics.csv").read_bytes()
        ).hexdigest(),
        "validation_cutoff_utc": pd.to_datetime(VALIDATION_CUTOFF, unit="s", utc=True).isoformat(),
        "validation_end_utc": pd.to_datetime(TEST_CUTOFF, unit="s", utc=True).isoformat(),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": SPLIT_SEED,
        "interval": "95% paired user percentile bootstrap, conditional on this snapshot",
        "multiple_comparison_adjustment": "none; exploratory comparisons",
        "subgroups": "Same users for every method within each history, grouped by positive reference seed availability.",
        "popularity": "Count of positive ratings by reference users strictly before the cutoff.",
        "coverage_denominator": len(snapshot.candidate_movies),
        "hidden_rated_share": "Recommendation slots rated before cutoff but hidden in the selected history.",
        "empty_subgroups": "Metrics are blank when users = 0.",
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return tables


def _validate_results(
    results: pd.DataFrame, snapshot: Snapshot
) -> dict[tuple[int, str, str], list[int]]:
    """Reject mismatched cohorts, invalid rankings, and stale hit calculations."""
    keys = ["user_id", "history_length", "method"]
    expected = pd.MultiIndex.from_product(
        [snapshot.target_users, HISTORIES, METHODS], names=keys
    )
    actual = pd.MultiIndex.from_frame(results[keys])
    if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError("Results must contain every cohort user once per history and method.")
    relevant = snapshot.held_out_positives.groupby("user_id")["movie_id"].apply(set)
    catalog = set(snapshot.candidate_movies)
    histories = {}
    seed_users = {}
    positive_movie_ids = set(snapshot.positive_edges["movie_id"])
    for length, name in zip(HISTORY_LENGTHS, HISTORIES):
        visible = snapshot.visible_history(length)
        histories[name] = visible.groupby("user_id")["movie_id"].apply(set)
        seed_users[name] = set(visible.loc[
            visible["rating"].ge(4) & visible["movie_id"].isin(positive_movie_ids), "user_id"
        ])
    rankings = {}
    valid_reasons = {"", "no_usable_seed", "no_positive_unseen_candidate"}
    for row in results.itertuples():
        movie_ids = json.loads(row.top_10_movie_ids)
        seen = histories[row.history_length][row.user_id]
        if not isinstance(movie_ids, list) or any(type(movie_id) is not int for movie_id in movie_ids):
            raise ValueError(f"Invalid movie list for user {row.user_id}.")
        hits = len(set(movie_ids) & relevant[row.user_id])
        no_seed = row.user_id not in seed_users[row.history_length]
        if (
            len(movie_ids) != TOP_K
            or len(set(movie_ids)) != TOP_K
            or not set(movie_ids).issubset(catalog)
            or bool(set(movie_ids) & seen)
            or row.visible_ratings != len(seen)
            or row.hit_count != hits
            or row.eligible_relevant_item_count != len(relevant[row.user_id])
            or not np.isclose(row.precision_at_10, hits / TOP_K)
            or not np.isclose(row.recall_at_10, hits / len(relevant[row.user_id]))
            or row.fallback_reason not in valid_reasons
            or (row.method == "popularity" and row.fallback_reason != "")
            or (row.method != "popularity" and no_seed != (row.fallback_reason == "no_usable_seed"))
        ):
            raise ValueError(f"Invalid saved recommendation row for user {row.user_id}.")
        rankings[(row.user_id, row.history_length, row.method)] = movie_ids
    for row in results.loc[results["fallback_reason"].ne("")].itertuples():
        if rankings[(row.user_id, row.history_length, row.method)] != rankings[
            (row.user_id, row.history_length, "popularity")
        ]:
            raise ValueError("Fallback recommendations must match popularity for the same user.")
    return rankings
