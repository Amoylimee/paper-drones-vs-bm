from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


def _bootstrap_local_packages() -> None:
    package_roots = [
        Path("/Users/jeremy/Downloads/my_packages/sinbue/src"),
        Path("/Users/jeremy/Downloads/my_packages/iogenius/src"),
        Path("/Users/jeremy/Downloads/my_packages/marenem/src"),
    ]
    for root in package_roots:
        if root.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    return root


def _build_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("P1_p3 Cleaned Data Inspection")
    lines.append("=" * 60)
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append(f"Time range: {df['timeUtc'].min()} -> {df['timeUtc'].max()}")
    lines.append(f"Unique MMSI: {df['mmsi'].nunique():,}")
    lines.append(
        "Duplicate records (mmsi + timeUtc): "
        f"{df.duplicated(subset=['mmsi', 'timeUtc']).sum():,}"
    )
    lines.append("")
    lines.append("Missing Values")
    lines.append("-" * 60)
    lines.append(df.isna().sum().to_string())
    lines.append("")
    lines.append("Dtypes")
    lines.append("-" * 60)
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("Numeric Describe")
    lines.append("-" * 60)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    lines.append(df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    _bootstrap_local_packages()
    import iogenius as iog

    root = _project_root()
    iog.set_working_directory(str(root))

    output_dir = Path("./output/P1/p3_cleaned_data_inspection")
    log_dir = Path("./logs/P1/p3_cleaned_data_inspection")
    iog.create_new_directory(output_dir)
    iog.create_new_directory(log_dir)

    cleaned_data_path = Path("./output/P1/p2_traj_cleaning/P1_p2_cleaned_traj.feather")
    df_cleaned = pd.read_feather(cleaned_data_path)
    df_cleaned["timeUtc"] = pd.to_datetime(df_cleaned["timeUtc"], errors="coerce")

    summary_text = _build_summary(df_cleaned)
    summary_path = output_dir / "P1_p3_cleaned_data_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    (log_dir / "p3_cleaned_data_inspection.log").write_text(
        f"Completed cleaned-data inspection. Summary saved to {summary_path}\n",
        encoding="utf-8",
    )
