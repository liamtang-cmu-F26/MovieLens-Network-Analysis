"""Figure generation for the preliminary network analysis."""

from __future__ import annotations

from pathlib import Path

from .utils import configure_matplotlib

configure_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .data import MovieLensData
from .utils import logarithmic_bins, shorten_title


def create_bipartite_figure(data: MovieLensData, output_dir: Path) -> None:
    """Create a readable, deterministic sample of the weighted graph."""

    ratings = data.ratings
    candidate_movies = ratings.groupby("movie_id").size().nlargest(24).index
    candidate_edges = ratings[ratings["movie_id"].isin(candidate_movies)]
    user_coverage = (
        candidate_edges.groupby("user_id").size().rename("coverage").reset_index()
        .sort_values(["coverage", "user_id"], ascending=[False, True])
    )
    selected_users = user_coverage.head(12)["user_id"]
    selected_edges = candidate_edges[candidate_edges["user_id"].isin(selected_users)]
    selected_movies = (
        selected_edges.groupby("movie_id").size().rename("coverage").reset_index()
        .sort_values(["coverage", "movie_id"], ascending=[False, True])
        .head(8)["movie_id"]
    )
    sample = selected_edges[selected_edges["movie_id"].isin(selected_movies)].copy()

    users = sorted(sample["user_id"].unique())
    movie_ids = sorted(sample["movie_id"].unique())
    user_y = dict(zip(users, np.linspace(0.95, 0.05, len(users))))
    movie_y = dict(zip(movie_ids, np.linspace(0.92, 0.08, len(movie_ids))))
    segments = [
        [(0.0, user_y[int(row.user_id)]), (1.0, movie_y[int(row.movie_id)])]
        for row in sample.itertuples(index=False)
    ]
    weights = sample["rating"].to_numpy(dtype=float)
    color_map = plt.get_cmap("RdYlGn")
    normalization = Normalize(vmin=1, vmax=5)

    figure, axis = plt.subplots(figsize=(13.5, 8.2), constrained_layout=True)
    axis.add_collection(
        LineCollection(
            segments,
            colors=color_map(normalization(weights)),
            linewidths=0.45 + weights * 0.28,
            alpha=0.55,
            zorder=1,
        )
    )
    axis.scatter(
        np.zeros(len(users)), [user_y[user] for user in users], s=95,
        color="#277DA1", edgecolor="white", linewidth=0.8, zorder=3,
    )
    axis.scatter(
        np.ones(len(movie_ids)), [movie_y[movie] for movie in movie_ids], s=150,
        color="#F8961E", edgecolor="white", linewidth=0.8, zorder=3,
    )

    for user_id in users:
        axis.text(-0.035, user_y[user_id], f"User {user_id}", ha="right", va="center", fontsize=9)
    titles = data.movies.set_index("movie_id")["title"].to_dict()
    for movie_id in movie_ids:
        axis.text(
            1.035, movie_y[movie_id], shorten_title(str(titles[movie_id])),
            ha="left", va="center", fontsize=9,
        )

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        ax=axis, location="bottom", shrink=0.38, pad=0.02, aspect=35,
    )
    colorbar.set_label("Rating (edge weight)")
    colorbar.set_ticks([1, 2, 3, 4, 5])
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#277DA1", markeredgecolor="white", markersize=9, label="User node"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#F8961E", markeredgecolor="white", markersize=10, label="Movie node"),
        ],
        loc="upper center", ncol=2, frameon=False,
    )
    axis.text(0, 1.015, "Users", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axis.text(1, 1.015, "Movies", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axis.set_title(
        "Figure 1. Illustrative sample of the weighted MovieLens bipartite network",
        fontsize=15, pad=30, fontweight="bold",
    )
    axis.text(
        0.5, -0.045,
        "Deterministic high-overlap sample; shown for readability, not as a representative sample of the full graph.",
        transform=axis.transAxes, ha="center", va="top", fontsize=9, color="#555555",
    )
    axis.set_xlim(-0.32, 1.45)
    axis.set_ylim(0, 1)
    axis.axis("off")

    _save_figure(figure, output_dir, "figure_1_bipartite_network")


def create_degree_figure(
    user_degree: pd.Series,
    movie_degree: pd.Series,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    _plot_degree_panel(axes[0, 0], axes[1, 0], user_degree, "#277DA1", "User")
    _plot_degree_panel(axes[0, 1], axes[1, 1], movie_degree, "#F8961E", "Movie")
    axes[0, 0].axvline(20, color="#D62828", linestyle="--", linewidth=1.5, label="Dataset minimum: 20")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].axvline(20, color="#D62828", linestyle="--", linewidth=1.5)
    figure.suptitle(
        "Figure 2. User and movie degree distributions in MovieLens 1M",
        fontsize=16, fontweight="bold",
    )
    figure.text(
        0.5, -0.01,
        "CCDF = complementary cumulative distribution. Log-log axes show tails; they do not establish a power law.",
        ha="center", fontsize=9, color="#555555",
    )
    _save_figure(figure, output_dir, "figure_2_degree_distributions")


def _plot_degree_panel(
    histogram_axis: plt.Axes,
    ccdf_axis: plt.Axes,
    degree: pd.Series,
    color: str,
    node_type: str,
) -> None:
    values = degree.to_numpy(dtype=int)
    histogram_axis.hist(
        values, bins=logarithmic_bins(values), color=color, edgecolor="white", linewidth=0.5
    )
    histogram_axis.set(xscale="log", yscale="log", xlabel="Degree", ylabel="Number of nodes")
    histogram_axis.set_title(f"{node_type} degree histogram", fontweight="bold")
    histogram_axis.grid(True, which="both", alpha=0.2)

    unique_degree, counts = np.unique(values, return_counts=True)
    ccdf = np.cumsum(counts[::-1])[::-1] / len(values)
    ccdf_axis.step(unique_degree, ccdf, where="post", color=color, linewidth=2)
    ccdf_axis.set(xscale="log", yscale="log", xlabel="Degree k", ylabel="P(K ≥ k)")
    ccdf_axis.set_title(f"{node_type} degree CCDF", fontweight="bold")
    ccdf_axis.grid(True, which="both", alpha=0.2)

    histogram_axis.text(
        0.97, 0.95,
        f"n = {len(values):,}\nmedian = {np.median(values):,.0f}\nmean = {np.mean(values):,.1f}\nmax = {np.max(values):,}",
        transform=histogram_axis.transAxes, ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )


def _save_figure(figure: plt.Figure, output_dir: Path, filename: str) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{filename}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)

