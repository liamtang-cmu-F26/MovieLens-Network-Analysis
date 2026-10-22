"""Command-line entry point for the Milestone 1 analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from movielens_network.analysis import analyze_network
from movielens_network.data import load_movielens
from movielens_network.plots import create_bipartite_figure, create_degree_figure
from movielens_network.reporting import write_summary_table


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Table 1 and Figures 1–2 for MovieLens 1M."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR / "ml-1m",
        help="Directory containing ratings.dat and movies.dat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs",
        help="Directory for generated tables and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_movielens(args.data_dir.resolve())
    result = analyze_network(data)

    write_summary_table(result.summary, output_dir)
    create_bipartite_figure(data, output_dir)
    create_degree_figure(result.user_degree, result.movie_degree, output_dir)

    print(f"Generated Table 1 and Figures 1–2 in {output_dir}")


if __name__ == "__main__":
    main()

