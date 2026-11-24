"""Community sizes and genre composition of the reference movie projection."""

from pathlib import Path

from .utils import configure_matplotlib

configure_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_community_figure(
    tables: dict[str, pd.DataFrame], metadata: dict[str, object], output_dir: Path,
) -> None:
    summary = tables["community_summary"].head(12)
    genres = tables["community_genres"].pivot(
        index="community_id", columns="genre", values="fraction_of_community"
    ).fillna(0).reindex(summary["community_id"])
    figure, axes = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 2.5]},
        constrained_layout=True,
    )
    y = np.arange(len(summary))
    axes[0].barh(y, summary["movies"], color="#237B9F")
    axes[0].set_yticks(y, [f"C{value}" for value in summary["community_id"]])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Movies", title="Community sizes")
    axes[0].bar_label(axes[0].containers[0], padding=3, fontsize=9)
    axes[0].set_xlim(0, summary["movies"].max() * 1.17)
    axes[0].grid(axis="x", alpha=0.2)
    heatmap = axes[1].imshow(genres.to_numpy() * 100, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    axes[1].set_yticks(y, [f"C{value}" for value in summary["community_id"]])
    axes[1].set_xticks(np.arange(len(genres.columns)), genres.columns, rotation=55, ha="right")
    axes[1].set_title("Movies with each genre (%)")
    figure.colorbar(heatmap, ax=axes[1], label="Within-community prevalence (%)", shrink=0.8)
    stability = tables["community_stability"].iloc[1:]["adjusted_rand_vs_seed_755"]
    figure.suptitle(
        f"{str(metadata['snapshot']).title()} reference movie communities\n"
        f"{metadata['connected_movies']:,} connected movies | "
        f"{metadata['communities']} communities | "
        f"weighted modularity {metadata['weighted_modularity']:.3f}",
        fontweight="bold", fontsize=14,
    )
    figure.supxlabel(
        f"Largest {len(summary)} communities shown. Genres may overlap. "
        f"Isolates excluded: {metadata['isolated_movies']}.\n"
        f"Seed agreement with 755 (adjusted Rand): {stability.min():.3f}–{stability.max():.3f}. "
        "Graph thresholds and genre patterns are descriptive.",
        fontsize=9,
    )
    figure.savefig(output_dir / "figure_movie_communities.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
