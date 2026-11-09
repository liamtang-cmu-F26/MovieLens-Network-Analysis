"""Global-time snapshots and simulated target histories for MovieLens 1M."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


VALIDATION_CUTOFF = int(pd.Timestamp("2000-12-02 14:52:18", tz="UTC").timestamp())
TEST_CUTOFF = int(pd.Timestamp("2000-12-29 23:43:34", tz="UTC").timestamp())
SPLIT_SEED = 755
REFERENCE_FRACTION = 0.70
MIN_HISTORY = 10
HISTORY_LENGTHS = (1, 5, 10, None)


@dataclass(frozen=True)
class Snapshot:
    """Reference inputs, target histories, and evaluator-only held-out labels."""

    reference_ratings: pd.DataFrame
    candidate_movies: pd.Index
    target_history: pd.DataFrame
    held_out_positives: pd.DataFrame
    target_users: pd.Index
    targets_with_min_history: int
    targets_with_later_positives: int
    later_positives: int
    positives_outside_catalog: int

    @property
    def positive_edges(self) -> pd.DataFrame:
        """Keep star values so scorers can choose binary or weighted edges."""
        return self.reference_ratings.loc[self.reference_ratings["rating"].ge(4)]

    def visible_history(self, length: int | None) -> pd.DataFrame:
        """Return observed ratings, including low ratings needed for seen masks."""
        if length is None:
            return self.target_history
        if length < 1:
            raise ValueError("History length must be positive.")
        return self.target_history.groupby("user_id", sort=False).head(length)


def select_target_users(ratings: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Index:
    """Reserve a fixed user-level holdout independent of input row order."""
    user_ids = np.sort(ratings["user_id"].unique())
    shuffled = np.random.default_rng(seed).permutation(user_ids)
    reference_count = int(REFERENCE_FRACTION * len(user_ids))
    return pd.Index(np.sort(shuffled[reference_count:]), name="user_id")


def build_snapshot(
    ratings: pd.DataFrame,
    target_users: pd.Index,
    cutoff: int,
    end: int | None = None,
) -> Snapshot:
    """Build a snapshot with a half-open evaluation window [cutoff, end)."""
    if end is not None and end <= cutoff:
        raise ValueError("Evaluation end must be after the cutoff.")

    is_target = ratings["user_id"].isin(target_users)
    is_before = ratings["timestamp"].lt(cutoff)
    reference = ratings.loc[~is_target & is_before]
    candidates = pd.Index(reference["movie_id"].unique(), name="movie_id").sort_values()

    history = ratings.loc[is_target & is_before]
    history_counts = history.groupby("user_id").size()
    users_with_history = history_counts.index[history_counts.ge(MIN_HISTORY)]

    in_window = ~is_before
    if end is not None:
        in_window &= ratings["timestamp"].lt(end)
    positives = ratings.loc[
        in_window & ratings["user_id"].isin(users_with_history) & ratings["rating"].ge(4)
    ]
    eligible = positives.loc[positives["movie_id"].isin(candidates)]
    cohort = pd.Index(eligible["user_id"].unique(), name="user_id").sort_values()
    target_history = history.loc[history["user_id"].isin(cohort)].sort_values(
        ["user_id", "timestamp", "movie_id"]
    )

    return Snapshot(
        reference_ratings=reference,
        candidate_movies=candidates,
        target_history=target_history,
        held_out_positives=eligible,
        target_users=cohort,
        targets_with_min_history=len(users_with_history),
        targets_with_later_positives=positives["user_id"].nunique(),
        later_positives=len(positives),
        positives_outside_catalog=len(positives) - len(eligible),
    )


def audit_snapshot(snapshot: Snapshot) -> list[dict[str, int | str]]:
    """Count data eligibility and graph seed availability before scoring."""
    positive_edges = snapshot.positive_edges
    seed_movies = pd.Index(positive_edges["movie_id"].unique())
    rows = []
    for length in HISTORY_LENGTHS:
        visible = snapshot.visible_history(length)
        positive = visible.loc[visible["rating"].ge(4)]
        usable = positive.loc[positive["movie_id"].isin(seed_movies)]
        users_with_seed = usable["user_id"].nunique()
        rows.append({
            "history_length": "full" if length is None else length,
            "reference_users": snapshot.reference_ratings["user_id"].nunique(),
            "reference_ratings": len(snapshot.reference_ratings),
            "reference_positive_edges": len(positive_edges),
            "candidate_movies": len(snapshot.candidate_movies),
            "candidate_movies_with_positive_degree": len(seed_movies),
            "targets_with_min_history": snapshot.targets_with_min_history,
            "targets_with_later_positives": snapshot.targets_with_later_positives,
            "cohort_users": len(snapshot.target_users),
            "later_positives_before_catalog_filter": snapshot.later_positives,
            "positives_outside_catalog": snapshot.positives_outside_catalog,
            "eligible_positives": len(snapshot.held_out_positives),
            "visible_ratings": len(visible),
            "users_with_visible_positive": positive["user_id"].nunique(),
            "users_with_usable_seed": users_with_seed,
            "users_without_usable_seed": len(snapshot.target_users) - users_with_seed,
        })
    return rows
