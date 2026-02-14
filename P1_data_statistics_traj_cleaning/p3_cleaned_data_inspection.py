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
    speed_col = "speed" if "speed" in df.columns else "sog"
    avg_speed = df[speed_col].mean(skipna=True) if speed_col in df.columns else float("nan")
    valid_speed_count = (
        df[speed_col].notna().sum() if speed_col in df.columns else 0
    )
    unique_id_count = df["id"].nunique() if "id" in df.columns else 0

    lines: list[str] = []
    lines.append("P1_p3 Cleaned AIS Fleet Report")
    lines.append("=" * 60)
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append(f"Time range: {df['timeUtc'].min()} -> {df['timeUtc'].max()}")
    lines.append(f"Unique MMSI: {df['mmsi'].nunique():,}")
    lines.append(f"Unique ID: {unique_id_count:,}")
    lines.append(f"Average Fleet Speed ({speed_col}): {avg_speed:.3f}")
    lines.append(f"Speed observations used: {valid_speed_count:,}")
    lines.append(
        "Duplicate records (mmsi + timeUtc): "
        f"{df.duplicated(subset=['mmsi', 'timeUtc']).sum():,}"
    )
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
