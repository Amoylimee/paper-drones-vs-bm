from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_COLUMNS


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


def _load_raw_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, header=None, names=RAW_DATA_COLUMNS)
    df["timeUtc"] = pd.to_datetime(df["timeUtc"], errors="coerce")
    return df


def _build_summary(df: pd.DataFrame) -> str:
    speed_col = "speed" if "speed" in df.columns else "sog"
    avg_speed = df[speed_col].mean(skipna=True) if speed_col in df.columns else float("nan")
    valid_speed_count = (
        df[speed_col].notna().sum() if speed_col in df.columns else 0
    )
    unique_id_count = df["id"].nunique() if "id" in df.columns else 0
    interval_stats = (
        df.sort_values(["mmsi", "timeUtc"])
        .groupby("mmsi")["timeUtc"]
        .diff()
        .dt.total_seconds()
    )
    interval_stats = interval_stats[(interval_stats.notna()) & (interval_stats > 0)]
    avg_interval_seconds = interval_stats.mean(skipna=True)
    valid_interval_count = interval_stats.shape[0]

    lines: list[str] = []
    lines.append("P1_p1 Raw AIS Fleet Report")
    lines.append("=" * 60)
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append(f"Time range: {df['timeUtc'].min()} -> {df['timeUtc'].max()}")
    lines.append(f"Unique MMSI: {df['mmsi'].nunique():,}")
    lines.append(f"Unique ID: {unique_id_count:,}")
    lines.append(f"Average Fleet Speed ({speed_col}): {avg_speed:.3f}")
    lines.append(f"Speed observations used: {valid_speed_count:,}")
    lines.append(f"Average Interval Between Points (seconds): {avg_interval_seconds:.3f}")
    lines.append(f"Interval observations used: {valid_interval_count:,}")
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

    output_dir = Path("./output/P1/p1_raw_data_inspection")
    log_dir = Path("./logs/P1/p1_raw_data_inspection")
    iog.create_new_directory(output_dir)
    iog.create_new_directory(log_dir)

    raw_data_path = Path("./data/bridge_msg_filtered.csv")
    df_raw = _load_raw_data(raw_data_path)

    summary_text = _build_summary(df_raw)
    summary_path = output_dir / "P1_p1_raw_data_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    (log_dir / "p1_raw_data_inspection.log").write_text(
        f"Completed raw inspection. Summary saved to {summary_path}\n",
        encoding="utf-8",
    )
