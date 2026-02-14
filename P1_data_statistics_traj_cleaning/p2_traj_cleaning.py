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


def _compute_interval_table(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["mmsi", "timeUtc"]].dropna(subset=["mmsi", "timeUtc"]).copy()
    data = data.sort_values(["mmsi", "timeUtc"]).reset_index(drop=True)
    data["interval_seconds"] = data.groupby("mmsi")["timeUtc"].diff().dt.total_seconds()
    data = data[data["interval_seconds"].notna() & (data["interval_seconds"] > 0)]

    if data.empty:
        return pd.DataFrame(
            columns=[
                "mmsi",
                "interval_count",
                "avg_interval_seconds",
                "median_interval_seconds",
                "p90_interval_seconds",
                "avg_frequency_hz",
            ]
        )

    out = (
        data.groupby("mmsi")["interval_seconds"]
        .agg(
            interval_count="count",
            avg_interval_seconds="mean",
            median_interval_seconds="median",
            p90_interval_seconds=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    out["avg_frequency_hz"] = 1.0 / out["avg_interval_seconds"]
    return out.sort_values("avg_interval_seconds").reset_index(drop=True)


def _build_mmsi_row_change_table(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> pd.DataFrame:
    raw_counts = df_raw.groupby("mmsi").size().rename("raw_rows")
    cleaned_counts = df_cleaned.groupby("mmsi").size().rename("cleaned_rows")
    out = pd.concat([raw_counts, cleaned_counts], axis=1).fillna(0)
    out[["raw_rows", "cleaned_rows"]] = out[["raw_rows", "cleaned_rows"]].astype(int)
    out = out.reset_index()
    out["rows_removed"] = out["raw_rows"] - out["cleaned_rows"]
    out["removed_ratio_pct"] = (
        out["rows_removed"] / out["raw_rows"].replace(0, pd.NA) * 100.0
    ).fillna(0.0)
    out = out.sort_values(["rows_removed", "raw_rows"], ascending=[False, False]).reset_index(drop=True)
    return out


def _format_diagnostics_report(
    step_df: pd.DataFrame,
    mmsi_row_change_df: pd.DataFrame,
    raw_interval_df: pd.DataFrame,
    cleaned_interval_df: pd.DataFrame,
) -> str:
    total_raw = int(step_df.loc[step_df["step"] == "raw_input", "rows_after"].iloc[0])
    total_cleaned = int(step_df.iloc[-1]["rows_after"])
    total_removed = total_raw - total_cleaned
    total_removed_pct = (total_removed / total_raw * 100.0) if total_raw else 0.0

    avg_interval_raw = raw_interval_df["avg_interval_seconds"].mean() if not raw_interval_df.empty else float("nan")
    avg_interval_cleaned = (
        cleaned_interval_df["avg_interval_seconds"].mean() if not cleaned_interval_df.empty else float("nan")
    )

    fully_removed_ships = int((mmsi_row_change_df["cleaned_rows"] == 0).sum())
    ships_50_removed = int((mmsi_row_change_df["removed_ratio_pct"] >= 50).sum())
    ships_80_removed = int((mmsi_row_change_df["removed_ratio_pct"] >= 80).sum())

    lines: list[str] = []
    lines.append("P1_p2 Trajectory Cleaning Report")
    lines.append("=" * 60)
    lines.append(f"Raw rows: {total_raw:,}")
    lines.append(f"Cleaned rows: {total_cleaned:,}")
    lines.append(f"Total rows removed: {total_removed:,} ({total_removed_pct:.2f}%)")
    lines.append("")
    lines.append("Per-MMSI Average Interval (seconds)")
    lines.append("-" * 60)
    lines.append(f"Raw (mean of vessel means): {avg_interval_raw:.3f}")
    lines.append(f"Cleaned (mean of vessel means): {avg_interval_cleaned:.3f}")
    lines.append("")
    lines.append("Step-by-Step Removal")
    lines.append("-" * 60)
    lines.append(step_df.to_string(index=False))
    lines.append("")
    lines.append("Per-MMSI Impact Summary")
    lines.append("-" * 60)
    lines.append(f"Total MMSI in raw: {mmsi_row_change_df['mmsi'].nunique():,}")
    lines.append(f"MMSI fully removed (0 rows left): {fully_removed_ships:,}")
    lines.append(f"MMSI with >=50% rows removed: {ships_50_removed:,}")
    lines.append(f"MMSI with >=80% rows removed: {ships_80_removed:,}")
    lines.append("")
    lines.append("Top 20 MMSI by rows removed")
    lines.append("-" * 60)
    lines.append(mmsi_row_change_df.head(20).to_string(index=False))
    return "\n".join(lines) + "\n"


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

        steps = [
            "basic",
            "tbd_drift",
            # "bidirectional_speed_drift",
            "speed_anomalies",
        ]
        step_records: list[dict[str, float | str | int]] = []
        initial_rows = len(cleaner.data)
        step_records.append(
            {
                "step": "raw_input",
                "rows_before": initial_rows,
                "rows_after": initial_rows,
                "rows_removed": 0,
                "removed_pct_of_previous": 0.0,
                "removed_pct_of_initial": 0.0,
            }
        )

        for step in steps:
            rows_before = len(cleaner.data)
            cleaner.clean(method=step)
            rows_after = len(cleaner.data)
            rows_removed = rows_before - rows_after
            step_records.append(
                {
                    "step": step,
                    "rows_before": rows_before,
                    "rows_after": rows_after,
                    "rows_removed": rows_removed,
                    "removed_pct_of_previous": (rows_removed / rows_before * 100.0)
                    if rows_before
                    else 0.0,
                    "removed_pct_of_initial": (rows_removed / initial_rows * 100.0)
                    if initial_rows
                    else 0.0,
                }
            )

        df_cleaned = cleaner.data.copy()
        df_cleaned["sog"] = df_cleaned["speed"]
        df_cleaned = df_cleaned.sort_values(["mmsi", "timeUtc"]).reset_index(drop=True)
        step_df = pd.DataFrame(step_records)
        raw_interval_df = _compute_interval_table(df_raw)
        cleaned_interval_df = _compute_interval_table(df_cleaned)
        mmsi_row_change_df = _build_mmsi_row_change_table(df_raw, df_cleaned)

        cleaned_feather_path = output_dir / "P1_p2_cleaned_traj.feather"
        df_cleaned.to_feather(cleaned_feather_path)

        cleaned_csv_path = output_dir / "P1_p2_cleaned_traj.csv"
        df_cleaned.to_csv(cleaned_csv_path, index=False)

        step_volume_path = output_dir / "P1_p2_step_volume_changes.csv"
        step_df.to_csv(step_volume_path, index=False)
        mmsi_change_path = output_dir / "P1_p2_mmsi_row_changes.csv"
        mmsi_row_change_df.to_csv(mmsi_change_path, index=False)
        mmsi_interval_raw_path = output_dir / "P1_p2_mmsi_intervals_raw.csv"
        raw_interval_df.to_csv(mmsi_interval_raw_path, index=False)
        mmsi_interval_cleaned_path = output_dir / "P1_p2_mmsi_intervals_cleaned.csv"
        cleaned_interval_df.to_csv(mmsi_interval_cleaned_path, index=False)
        report_text = _format_diagnostics_report(
            step_df=step_df,
            mmsi_row_change_df=mmsi_row_change_df,
            raw_interval_df=raw_interval_df,
            cleaned_interval_df=cleaned_interval_df,
        )
        report_path = output_dir / "P1_p2_cleaning_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        removed_rows = len(df_raw) - len(df_cleaned)
        avg_interval_cleaned = (
            cleaned_interval_df["avg_interval_seconds"].mean()
            if not cleaned_interval_df.empty
            else float("nan")
        )
        print(f"Cleaned rows: {len(df_cleaned):,}")
        print(f"Removed rows: {removed_rows:,}")
        print(f"Removed ratio: {removed_rows / len(df_raw) * 100:.2f}%")
        print(f"Per-MMSI avg interval (seconds, cleaned): {avg_interval_cleaned:.3f}")
        print(f"Saved feather: {cleaned_feather_path}")
        print(f"Saved csv: {cleaned_csv_path}")
        print(f"Saved report: {report_path}")
        print(f"Saved step table: {step_volume_path}")
        print(f"Saved mmsi row-change table: {mmsi_change_path}")
        print(f"Saved mmsi interval table(raw): {mmsi_interval_raw_path}")
        print(f"Saved mmsi interval table(cleaned): {mmsi_interval_cleaned_path}")
