"""Command-line entry point for Milestone 2 data and validation analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd

from movielens_network.data import load_movielens
from movielens_network.milestone2_data import (
    MIN_HISTORY,
    REFERENCE_FRACTION,
    SPLIT_SEED,
    TEST_CUTOFF,
    VALIDATION_CUTOFF,
    audit_snapshot,
    build_snapshot,
    select_target_users,
)
from movielens_network.milestone2_evaluation import evaluate_validation


ROOT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Write cohort counts for validation and test.")
    audit.add_argument("--data-dir", type=Path, default=ROOT_DIR / "ml-1m")
    audit.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "milestone2")
    evaluate = commands.add_parser(
        "evaluate", help="Run the three baseline recommenders on validation only."
    )
    evaluate.add_argument("--data-dir", type=Path, default=ROOT_DIR / "ml-1m")
    evaluate.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "milestone2")
    report = commands.add_parser("report", help="Analyze saved validation results and generate figures.")
    report.add_argument("--data-dir", type=Path, default=ROOT_DIR / "ml-1m")
    report.add_argument("--results-dir", type=Path, default=ROOT_DIR / "outputs" / "milestone2")
    report.add_argument(
        "--output-dir", type=Path,
        default=ROOT_DIR / "outputs" / "milestone2" / "analysis",
    )
    communities = commands.add_parser(
        "communities", help="Export reference movie graph and community analysis."
    )
    communities.add_argument("--data-dir", type=Path, default=ROOT_DIR / "ml-1m")
    communities.add_argument("--snapshot", choices=("validation", "test"), default="validation")
    communities.add_argument(
        "--output-dir", type=Path, default=ROOT_DIR / "outputs" / "milestone2" / "communities"
    )
    args = parser.parse_args()

    data = load_movielens(args.data_dir)
    if args.command == "communities":
        from movielens_network.milestone2_communities import analyze_communities

        cutoff = VALIDATION_CUTOFF if args.snapshot == "validation" else TEST_CUTOFF
        destination = args.output_dir / args.snapshot
        metadata = analyze_communities(data, cutoff, args.snapshot, destination)
        print(f"Exported {metadata['candidate_movies']} movies and {metadata['undirected_edges']} edges.")
        print(f"Found {metadata['communities']} communities. Outputs written to {destination.resolve()}")
        return

    ratings = data.ratings
    target_users = select_target_users(ratings)
    if args.command == "report":
        from movielens_network.milestone2_analysis import analyze_validation
        from movielens_network.milestone2_plots import create_validation_figures

        snapshot = build_snapshot(ratings, target_users, VALIDATION_CUTOFF, TEST_CUTOFF)
        tables = analyze_validation(snapshot, args.results_dir, args.output_dir)
        create_validation_figures(tables, args.output_dir)
        print(f"Validated saved recommendations for {len(snapshot.target_users)} users.")
        print(f"Four analysis tables and four figures written to {args.output_dir.resolve()}")
        return

    if args.command == "evaluate":
        snapshot = build_snapshot(
            ratings, target_users, VALIDATION_CUTOFF, TEST_CUTOFF
        )
        result = evaluate_validation(snapshot)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / "validation_metrics.csv"
        per_user_path = args.output_dir / "validation_per_user.csv"
        result.metrics.to_csv(metrics_path, index=False, float_format="%.10f")
        result.per_user.to_csv(per_user_path, index=False)
        print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.4%}"))
        print(f"Metrics written to {metrics_path.resolve()}")
        print(f"Per-user results written to {per_user_path.resolve()}")
        return

    windows = (
        ("validation", VALIDATION_CUTOFF, TEST_CUTOFF),
        ("test", TEST_CUTOFF, None),
    )
    rows = []
    for name, cutoff, end in windows:
        snapshot = build_snapshot(ratings, target_users, cutoff, end)
        rows.extend({"snapshot": name, **row} for row in audit_snapshot(snapshot))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame(rows)
    counts.to_csv(args.output_dir / "cohort_audit.csv", index=False)
    config = {
        "validation_cutoff_utc": pd.to_datetime(VALIDATION_CUTOFF, unit="s", utc=True).isoformat(),
        "test_cutoff_utc": pd.to_datetime(TEST_CUTOFF, unit="s", utc=True).isoformat(),
        "reference_fraction": REFERENCE_FRACTION,
        "split_seed": SPLIT_SEED,
        "assigned_target_users": len(target_users),
        "minimum_target_history": MIN_HISTORY,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(counts[[
        "snapshot", "history_length", "cohort_users", "eligible_positives",
        "users_with_usable_seed", "users_without_usable_seed",
    ]].to_string(index=False))
    print(f"Audit written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
