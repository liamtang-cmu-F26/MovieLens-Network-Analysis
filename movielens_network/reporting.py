"""Table 1 export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_summary_table(summary: pd.DataFrame, output_dir: Path) -> None:
    summary.to_csv(output_dir / "table_1_network_summary.csv", index=False)

    lines = [
        "# Table 1. MovieLens 1M weighted bipartite network summary",
        "",
        "| Section | Metric | Value | Unit | Note |",
        "|---|---|---:|---|---|",
    ]
    for row in summary.itertuples(index=False):
        note = str(row.note).replace("|", "\\|")
        lines.append(
            f"| {row.section} | {row.metric} | {_format_value(row.value, row.unit)} | "
            f"{row.unit} | {note} |"
        )
    lines.append("")
    (output_dir / "table_1_network_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _format_value(value: object, unit: str) -> str:
    if isinstance(value, (float, np.floating)):
        if unit == "fraction":
            return f"{100 * float(value):.3f}%"
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)

