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
    df["speed"] = df["sog"]
    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    _bootstrap_local_packages()
    import iogenius as iog
    import sinbue as sb

    root = _project_root()
    iog.set_working_directory(str(root))

    output_dir = Path("./output/P1/p2_traj_cleaning")
    log_dir = Path("./logs/P1/p2_traj_cleaning")
    iog.create_new_directory(output_dir)
    iog.create_new_directory(log_dir)

    log_file = log_dir / "p2_traj_cleaning.log"
    with sb.PrintRedirector(log_file):
        raw_data_path = Path("./data/bridge_msg_filtered.csv")
        df_raw = _load_raw_data(raw_data_path)
        print(f"Input rows: {len(df_raw):,}")

        cleaner = sb.AISCleanGenius(
            data=df_raw,
            cols=["mmsi", "timeUtc", "lon", "lat", "speed"],
        )

        cleaner.clean(method="basic")
        cleaner.clean(method="tbd_drift")
        cleaner.clean(method="bidirectional_speed_drift")
        cleaner.clean(method="speed_anomalies")

        df_cleaned = cleaner.data.copy()
        df_cleaned["sog"] = df_cleaned["speed"]
        df_cleaned = df_cleaned.sort_values(["mmsi", "timeUtc"]).reset_index(drop=True)

        cleaned_feather_path = output_dir / "P1_p2_cleaned_traj.feather"
        df_cleaned.to_feather(cleaned_feather_path)

        cleaned_csv_path = output_dir / "P1_p2_cleaned_traj.csv"
        df_cleaned.to_csv(cleaned_csv_path, index=False)

        removed_rows = len(df_raw) - len(df_cleaned)
        print(f"Cleaned rows: {len(df_cleaned):,}")
        print(f"Removed rows: {removed_rows:,}")
        print(f"Saved feather: {cleaned_feather_path}")
        print(f"Saved csv: {cleaned_csv_path}")
