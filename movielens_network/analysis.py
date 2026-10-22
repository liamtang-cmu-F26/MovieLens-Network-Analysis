"""Network statistics for the weighted user–movie graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MovieLensData


@dataclass(frozen=True)
class AnalysisResult:
    summary: pd.DataFrame
    user_degree: pd.Series
    movie_degree: pd.Series


class DisjointSet:
    """Union-find structure used to calculate connected components."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.component_size = np.ones(size, dtype=np.int32)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            next_item = int(self.parent[item])
            self.parent[item] = root
            item = next_item
        return root

    def union(self, first: int, second: int) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if self.component_size[root_a] < self.component_size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.component_size[root_a] += self.component_size[root_b]


def analyze_network(data: MovieLensData) -> AnalysisResult:
    ratings = data.ratings
    user_degree = ratings.groupby("user_id", sort=True).size().rename("degree")
    movie_degree = ratings.groupby("movie_id", sort=True).size().rename("degree")
    user_ids = user_degree.index.to_numpy(dtype=np.int32)
    movie_ids = movie_degree.index.to_numpy(dtype=np.int32)
    components = _component_statistics(ratings, user_ids, movie_ids)

    start = pd.to_datetime(ratings["timestamp"].min(), unit="s", utc=True)
    end = pd.to_datetime(ratings["timestamp"].max(), unit="s", utc=True)
    possible_edges = len(user_ids) * len(movie_ids)
    rows: list[dict[str, object]] = []

    def add(section: str, metric: str, value: object, unit: str, note: str = "") -> None:
        rows.append(
            {"section": section, "metric": metric, "value": value, "unit": unit, "note": note}
        )

    add("Network scale", "User nodes", len(user_ids), "nodes")
    add("Network scale", "Catalog movies", len(data.movies), "movies", "Rows in movies.dat")
    add("Network scale", "Rated movie nodes", len(movie_ids), "nodes", "Movies with at least one rating")
    add("Network scale", "Unrated catalog movies", len(data.movies) - len(movie_ids), "movies")
    add("Network scale", "Rating edges", len(ratings), "edges")
    add("Network scale", "Bipartite density", len(ratings) / possible_edges, "fraction")

    add("Connectivity", "Connected components", components["component_count"], "components")
    add("Connectivity", "Giant component nodes", components["giant_nodes"], "nodes")
    add("Connectivity", "Giant component share of nodes", components["giant_node_fraction"], "fraction")
    add("Connectivity", "Giant component share of users", components["giant_user_fraction"], "fraction")
    add("Connectivity", "Giant component share of movies", components["giant_movie_fraction"], "fraction")
    add("Connectivity", "Giant component share of edges", components["giant_edge_fraction"], "fraction")

    for label, value in _degree_statistics(user_degree).items():
        add("User degree", label, value, "movies rated per user")
    for label, value in _degree_statistics(movie_degree).items():
        add("Movie degree", label, value, "users per movie")

    add("Edge weights", "Minimum rating", int(ratings["rating"].min()), "stars")
    add("Edge weights", "Mean rating", float(ratings["rating"].mean()), "stars")
    add("Edge weights", "Maximum rating", int(ratings["rating"].max()), "stars")
    add(
        "Edge weights",
        "Ratings of 4 or 5",
        float(ratings["rating"].ge(4).mean()),
        "fraction",
        "Relevant-item threshold from the proposal",
    )
    add("Time", "First rating", start.strftime("%Y-%m-%d %H:%M:%S UTC"), "timestamp")
    add("Time", "Last rating", end.strftime("%Y-%m-%d %H:%M:%S UTC"), "timestamp")

    return AnalysisResult(pd.DataFrame(rows), user_degree, movie_degree)


def _degree_statistics(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    return {
        "Minimum": float(np.min(array)),
        "Median": float(np.median(array)),
        "Mean": float(np.mean(array)),
        "90th percentile": float(np.percentile(array, 90)),
        "Maximum": float(np.max(array)),
    }


def _component_statistics(
    ratings: pd.DataFrame,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> dict[str, float]:
    user_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    movie_offset = len(user_ids)
    movie_index = {
        int(movie_id): movie_offset + index for index, movie_id in enumerate(movie_ids)
    }
    components = DisjointSet(len(user_ids) + len(movie_ids))

    for user_id, movie_id in ratings[["user_id", "movie_id"]].itertuples(index=False, name=None):
        components.union(user_index[int(user_id)], movie_index[int(movie_id)])

    roots = np.fromiter(
        (components.find(index) for index in range(len(components.parent))),
        dtype=np.int32,
        count=len(components.parent),
    )
    component_roots, component_sizes = np.unique(roots, return_counts=True)
    giant_root = int(component_roots[int(np.argmax(component_sizes))])
    giant_users = user_ids[roots[: len(user_ids)] == giant_root]
    giant_movies = movie_ids[roots[len(user_ids) :] == giant_root]

    return {
        "component_count": int(len(component_roots)),
        "giant_nodes": int(np.max(component_sizes)),
        "giant_node_fraction": float(np.max(component_sizes) / len(roots)),
        "giant_user_fraction": float(len(giant_users) / len(user_ids)),
        "giant_movie_fraction": float(len(giant_movies) / len(movie_ids)),
        "giant_edge_fraction": float(ratings["user_id"].isin(giant_users).mean()),
    }

