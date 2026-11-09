"""Command-line entry point for the Milestone 2 data audit."""

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


ROOT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Write cohort counts for validation and test.")
    audit.add_argument("--data-dir", type=Path, default=ROOT_DIR / "ml-1m")
    audit.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "milestone2")
    args = parser.parse_args()

    ratings = load_movielens(args.data_dir).ratings
    target_users = select_target_users(ratings)
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
