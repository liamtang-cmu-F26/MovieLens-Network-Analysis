"""Reference-only movie projection, community detection, and node2vec exports."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .data import MovieLensData
from .milestone2_data import REFERENCE_FRACTION, SPLIT_SEED, select_target_users


NEIGHBORS = 20
MIN_SHARED_USERS = 3
COMMUNITY_SEEDS = (755, 756, 757)


def build_movie_graph(
    reference: pd.DataFrame, neighbors: int = NEIGHBORS,
    min_shared_users: int = MIN_SHARED_USERS,
) -> tuple[nx.Graph, pd.Series]:
    """Keep the union of each movie's strongest supported cosine neighbours."""
    if neighbors < 1 or min_shared_users < 1:
        raise ValueError("Graph thresholds must be positive.")
    movies = pd.Index(sorted(reference["movie_id"].unique()), name="movie_id")
    positive = reference.loc[reference["rating"].ge(4)]
    users = pd.Index(sorted(positive["user_id"].unique()))
    incidence = sparse.csr_matrix(
        (np.ones(len(positive)), (
            users.get_indexer(positive["user_id"]),
            movies.get_indexer(positive["movie_id"]),
        )),
        shape=(len(users), len(movies)),
    )
    degrees = np.asarray(incidence.sum(axis=0)).ravel()
    common = (incidence.T @ incidence).tocsr()
    graph = nx.Graph()
    graph.add_nodes_from(int(movie_id) for movie_id in movies)
    for col, movie_id in enumerate(movies):
        row = common.getrow(col)
        supported = (row.indices != col) & (row.data >= min_shared_users)
        other = row.indices[supported]
        counts = row.data[supported]
        weights = counts / np.sqrt(degrees[col] * degrees[other])
        order = np.lexsort((movies.to_numpy()[other], -weights))[:neighbors]
        for index in order:
            graph.add_edge(
                int(movie_id), int(movies[other[index]]),
                weight=float(weights[index]),
                shared_positive_users=int(counts[index]),
            )
    return graph, pd.Series(degrees.astype(int), index=movies, name="positive_reference_degree")


def adjusted_rand(first: np.ndarray, second: np.ndarray) -> float:
    """Compare partitions independently of their numeric community labels."""
    table = pd.crosstab(first, second).to_numpy(dtype=np.float64)
    pair_count = lambda values: np.sum(values * (values - 1) / 2)
    total_pairs = len(first) * (len(first) - 1) / 2
    if total_pairs == 0:
        return 1.0
    shared = pair_count(table)
    row_pairs = pair_count(table.sum(axis=1))
    col_pairs = pair_count(table.sum(axis=0))
    expected = row_pairs * col_pairs / total_pairs
    maximum = (row_pairs + col_pairs) / 2
    if maximum == expected:
        return 1.0
    return float((shared - expected) / (maximum - expected))


def analyze_communities(
    data: MovieLensData, cutoff: int, snapshot_name: str, output_dir: Path,
) -> dict[str, object]:
    """Construct a graph without using target histories or held-out ratings."""
    targets = select_target_users(data.ratings)
    reference = data.ratings.loc[
        ~data.ratings["user_id"].isin(targets) & data.ratings["timestamp"].lt(cutoff)
    ].sort_values(["user_id", "movie_id"])
    graph, positive_degree = build_movie_graph(reference)
    active_nodes = sorted(node for node, degree in graph.degree() if degree > 0)
    active = graph.subgraph(active_nodes).copy()
    if not active.number_of_edges():
        raise ValueError("No supported movie edges at this cutoff.")

    partitions = []
    stability_rows = []
    for seed in COMMUNITY_SEEDS:
        partition = nx.community.louvain_communities(
            active, weight="weight", resolution=1, threshold=1e-7, seed=seed,
        )
        partition = sorted(partition, key=lambda group: (-len(group), min(group)))
        labels = {node: index for index, group in enumerate(partition) for node in group}
        partitions.append(labels)
        stability_rows.append({
            "seed": seed,
            "communities": len(partition),
            "weighted_modularity": nx.community.modularity(active, partition, weight="weight"),
            "adjusted_rand_vs_seed_755": adjusted_rand(
                np.array([partitions[0][node] for node in active_nodes]),
                np.array([labels[node] for node in active_nodes]),
            ),
        })
    labels = partitions[0]
    nodes = data.movies.set_index("movie_id").loc[list(graph.nodes), ["title", "genres"]].copy()
    nodes["positive_reference_degree"] = positive_degree
    nodes["graph_degree"] = pd.Series(dict(graph.degree()))
    nodes["graph_strength"] = pd.Series(dict(graph.degree(weight="weight")))
    nodes["community_id"] = [labels.get(node, -1) for node in nodes.index]
    nodes["has_graph_edges"] = nodes["graph_degree"].gt(0)
    nodes = nodes.sort_index().reset_index()
    edge_rows = [
        {"source": min(first, second), "target": max(first, second), **attributes}
        for first, second, attributes in graph.edges(data=True)
    ]
    edges = pd.DataFrame(edge_rows).sort_values(["source", "target"])

    summary_rows = []
    genre_rows = []
    for community_id, members in nodes.loc[nodes["community_id"].ge(0)].groupby("community_id"):
        genre_counts = members["genres"].str.split("|").explode().value_counts()
        genre_counts = genre_counts.rename_axis("genre").reset_index(name="movies")
        genre_counts = genre_counts.sort_values(["movies", "genre"], ascending=[False, True])
        for row in genre_counts.itertuples(index=False):
            genre_rows.append({
                "community_id": community_id, "genre": row.genre,
                "movies": row.movies, "fraction_of_community": row.movies / len(members),
            })
        member_ids = set(members["movie_id"])
        internal_weight = active.subgraph(member_ids).size(weight="weight")
        representatives = members.sort_values(
            ["graph_strength", "movie_id"], ascending=[False, True]
        ).head(3)["title"]
        summary_rows.append({
            "community_id": community_id,
            "movies": len(members),
            "share_of_connected_movies": len(members) / len(active_nodes),
            "mean_positive_reference_degree": members["positive_reference_degree"].mean(),
            "internal_strength_share": 2 * internal_weight / members["graph_strength"].sum(),
            "top_genres": " | ".join(
                f"{row.genre} ({row.movies / len(members):.0%})"
                for row in genre_counts.head(3).itertuples(index=False)
            ),
            "representative_movies": " | ".join(representatives),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "movie_graph_nodes": nodes,
        "movie_graph_edges": edges,
        "community_summary": pd.DataFrame(summary_rows),
        "community_genres": pd.DataFrame(genre_rows),
        "community_stability": pd.DataFrame(stability_rows),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g")
    versions = {
        "python": platform.python_version(), "networkx": nx.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    metadata = {
        "snapshot": snapshot_name,
        "cutoff_utc": pd.to_datetime(cutoff, unit="s", utc=True).isoformat(),
        "reference_fraction": REFERENCE_FRACTION, "user_split_seed": SPLIT_SEED,
        "reference_ratings": len(reference),
        "reference_users": reference["user_id"].nunique(),
        "positive_reference_edges": int(positive_degree.sum()),
        "reference_ratings_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(reference, index=False).values.tobytes()
        ).hexdigest(),
        "candidate_movies": len(graph), "connected_movies": len(active_nodes),
        "isolated_movies": len(graph) - len(active_nodes),
        "undirected_edges": graph.number_of_edges(),
        "components_including_isolates": nx.number_connected_components(graph),
        "largest_component_movies": max(map(len, nx.connected_components(graph))),
        "neighbors_per_movie_before_union": NEIGHBORS,
        "minimum_shared_positive_users": MIN_SHARED_USERS,
        "edge_weight": "binary positive rating cosine similarity",
        "symmetrization": "union of top neighbours, each undirected edge exported once",
        "community_algorithm": "weighted Louvain",
        "community_resolution": 1, "community_threshold": 1e-7,
        "reported_partition_seed": COMMUNITY_SEEDS[0],
        "stability_seeds": list(COMMUNITY_SEEDS),
        "communities": len(summary_rows),
        "weighted_modularity": stability_rows[0]["weighted_modularity"],
        "versions": versions,
        "isolates": "community_id=-1, retained in nodes, excluded from Louvain and stability",
        "representatives": "three movies with highest weighted graph strength per community",
        "genre_interpretation": "multi-label genre prevalence, descriptive, no significance test",
        "stability_interpretation": "algorithm seeds on one fixed graph, not sampling uncertainty",
    }
    for name in ("movie_graph_nodes", "movie_graph_edges"):
        path = output_dir / f"{name}.csv"
        metadata[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "graph_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    from .milestone2_community_plots import create_community_figure

    create_community_figure(tables, metadata, output_dir)
    return metadata
