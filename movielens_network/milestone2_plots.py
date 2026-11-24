"""Figures from saved Milestone 2 validation analysis tables."""

from pathlib import Path

from .utils import configure_matplotlib

configure_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .milestone2_analysis import HISTORIES
from .milestone2_evaluation import METHODS


LABELS = {
    "popularity": "Popularity",
    "item_cosine": "Item cosine",
    "binary_diffusion": "Binary diffusion",
}
COLORS = {"popularity": "#687787", "item_cosine": "#237B9F", "binary_diffusion": "#C66A22"}
X = np.arange(len(HISTORIES))


def create_validation_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write four figures using the values exported in the analysis tables."""
    metrics = tables["validation_history_metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("precision_at_10", "recall_at_10"), ("Precision@10", "Recall@10")
    ):
        _method_lines(axis, metrics, metric, scale=100)
        axis.set(title=title, ylabel=f"{title} (%)")
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    figure.suptitle(
        f"Validation accuracy by visible history | {int(metrics.iloc[0]['users'])} users",
        fontweight="bold",
    )
    _save(figure, output_dir, "figure_1_history_length")

    differences = tables["validation_paired_differences"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    comparisons = differences[["first_method", "second_method"]].drop_duplicates()
    for axis, metric, title in zip(
        axes, ("precision_at_10", "recall_at_10"), ("Precision@10", "Recall@10")
    ):
        for offset, pair in enumerate(comparisons.itertuples(index=False)):
            group = differences.loc[
                differences["metric"].eq(metric)
                & differences["first_method"].eq(pair.first_method)
                & differences["second_method"].eq(pair.second_method)
            ].set_index("history_length").reindex(HISTORIES)
            center = group["mean_difference_pp"].to_numpy()
            bounds = np.vstack((
                center - group["ci_low_pp"].to_numpy(),
                group["ci_high_pp"].to_numpy() - center,
            ))
            axis.errorbar(
                X + (offset - 1) * 0.19, center, yerr=bounds,
                fmt=("o", "s", "^")[offset], capsize=3,
                color=("#237B9F", "#C66A22", "#72599B")[offset],
                label=f"{LABELS[pair.first_method]} − {LABELS[pair.second_method]}",
            )
        axis.axhline(0, color="#777777", linewidth=1)
        _history_axis(axis)
        axis.set(title=title, ylabel="Mean paired difference (percentage points)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Paired differences | 95% user bootstrap intervals", fontweight="bold")
    _save(figure, output_dir, "figure_2_paired_differences")

    subgroups = tables["validation_fallback_subgroups"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis, group_name, title in zip(
        axes[:2], ("usable_seed", "no_usable_seed"), ("Usable positive seed", "No usable positive seed")
    ):
        selected = subgroups.loc[subgroups["subgroup"].eq(group_name)]
        _method_lines(axis, selected, "precision_at_10", scale=100)
        sizes = selected.loc[selected["method"].eq("popularity")].set_index(
            "history_length"
        ).reindex(HISTORIES)["users"]
        axis.set_xticks(X, [f"{h.title()}\nn={int(n)}" for h, n in zip(HISTORIES, sizes)])
        axis.set(title=title, ylabel="Precision@10 (%)")
        axis.set_ylim(0, max(13, metrics["precision_at_10"].max() * 110))
    _method_lines(
        axes[2], metrics.assign(fallback_share=metrics["fallback_users"] / metrics["users"]),
        "fallback_share", scale=100,
    )
    axes[2].set(title="Recorded fallback use", ylabel="Users using fallback (%)", ylim=(-1, 100))
    axes[1].text(
        0.04, 0.94, "Empty groups have n=0",
        transform=axes[1].transAxes, va="top", fontsize=9, color="#555555",
    )
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Seed availability and fallback | subgroup membership varies with history", fontweight="bold")
    _save(figure, output_dir, "figure_3_fallback_subgroups")

    popularity = tables["validation_recommendation_popularity"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis, column, scale, title, ylabel in (
        (axes[0], "mean_positive_reference_degree", 1, "Recommendation popularity", "Mean positive reference degree"),
        (axes[1], "catalog_coverage", 100, "Catalog coverage", "Candidate movies recommended (%)"),
        (axes[2], "hidden_pre_cutoff_rated_share", 100, "Hidden earlier ratings", "Recommendation slots rated earlier (%)"),
    ):
        _method_lines(axis, popularity, column, scale=scale)
        axis.set(title=title, ylabel=ylabel)
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Recommendation exposure | popularity uses pre-cutoff reference ratings", fontweight="bold")
    _save(figure, output_dir, "figure_4_recommendation_popularity")


def _method_lines(axis: plt.Axes, table: pd.DataFrame, column: str, scale: float) -> None:
    for method in METHODS:
        values = table.loc[table["method"].eq(method)].set_index("history_length").reindex(HISTORIES)
        axis.plot(
            X, values[column] * scale, marker="o", linewidth=1.7,
            color=COLORS[method], label=LABELS[method],
        )
    _history_axis(axis)


def _history_axis(axis: plt.Axes) -> None:
    axis.set_xticks(X, [name.title() for name in HISTORIES])
    axis.set_xlabel("Visible ratings")
    axis.grid(axis="y", alpha=0.22)


def _save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
