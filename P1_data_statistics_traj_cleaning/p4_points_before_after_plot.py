from __future__ import annotations

import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io.img_tiles import MapboxTiles
from dotenv import load_dotenv
from matplotlib.collections import PatchCollection

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


def _load_cleaned_data(feather_path: Path, csv_path: Path) -> pd.DataFrame:
    if feather_path.exists():
        df = pd.read_feather(feather_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Cleaned data not found. Please run p2_traj_cleaning.py first."
        )
    df["timeUtc"] = pd.to_datetime(df["timeUtc"], errors="coerce")
    return df


def _get_dynamic_extent(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    q_low: float = 0.01,
    q_high: float = 0.99,
    pad_ratio: float = 0.06,
) -> tuple[float, float, float, float]:
    lon_all = pd.concat([df_raw[lon_col], df_cleaned[lon_col]], axis=0).dropna()
    lat_all = pd.concat([df_raw[lat_col], df_cleaned[lat_col]], axis=0).dropna()

    if lon_all.empty or lat_all.empty:
        raise ValueError("Cannot determine dynamic extent: missing lon/lat values.")

    lon_min = float(lon_all.quantile(q_low))
    lon_max = float(lon_all.quantile(q_high))
    lat_min = float(lat_all.quantile(q_low))
    lat_max = float(lat_all.quantile(q_high))

    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)

    lon_pad = lon_span * pad_ratio
    lat_pad = lat_span * pad_ratio

    return (
        lon_min - lon_pad,
        lon_max + lon_pad,
        lat_min - lat_pad,
        lat_max + lat_pad,
    )


def _set_plot_font() -> None:
    font_available = any("Times New Roman" in f.name for f in fm.fontManager.ttflist)
    if not font_available:
        font_dir = Path("fonts")
        if font_dir.exists():
            for font_file in font_dir.iterdir():
                if font_file.suffix.lower() in {".ttf", ".otf"}:
                    fm.fontManager.addfont(str(font_file))
    plt.rcParams["font.family"] = "Times New Roman"


def _points_to_grids(df: pd.DataFrame, boundary: tuple[float, float, float, float], resolution: float):
    import sinbue as sb

    data = df[["lon", "lat"]].dropna().copy()
    _, grids = sb.get_points_to_grids(
        data=data,
        cols=["lon", "lat"],
        boundary=[boundary[0], boundary[1], boundary[2], boundary[3]],
        resolution=resolution,
        mode="count",
    )
    return grids


def _grid_to_patches(grid_gdf) -> tuple[list, np.ndarray]:
    patches: list[mpatches.Polygon] = []
    values: list[float] = []

    if "point_count" not in grid_gdf.columns:
        return patches, np.array(values, dtype=float)

    for _, row in grid_gdf.iterrows():
        geom = row.geometry
        value = float(row["point_count"])

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            coords = np.asarray(geom.exterior.coords)
            patches.append(mpatches.Polygon(coords, closed=True))
            values.append(value)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                coords = np.asarray(part.exterior.coords)
                patches.append(mpatches.Polygon(coords, closed=True))
                values.append(value)

    return patches, np.array(values, dtype=float)


def _plot_before_after_grids(
    raw_grids,
    cleaned_grids,
    extent: tuple[float, float, float, float],
    save_path: Path,
    mapbox: MapboxTiles,
    basemap_zoom: int = 16,
    vmin_percentile: int = 0,
    vmax_percentile: int = 85,
) -> None:
    proj = ccrs.PlateCarree()
    lon_min, lon_max, lat_min, lat_max = extent

    raw_patches, raw_values = _grid_to_patches(raw_grids)
    cleaned_patches, cleaned_values = _grid_to_patches(cleaned_grids)

    all_values = []
    if len(raw_values) > 0:
        all_values.append(raw_values)
    if len(cleaned_values) > 0:
        all_values.append(cleaned_values)

    vmin, vmax = None, None
    if all_values:
        combined = np.concatenate(all_values)
        vmin = np.percentile(combined, vmin_percentile) if vmin_percentile > 0 else combined.min()
        vmax = np.percentile(combined, vmax_percentile)

    fig = plt.figure(figsize=(18, 8), dpi=300)
    ax1 = fig.add_axes([0.04, 0.12, 0.43, 0.80], projection=mapbox.crs)
    ax2 = fig.add_axes([0.51, 0.12, 0.43, 0.80], projection=mapbox.crs)
    cax = fig.add_axes([0.95, 0.12, 0.015, 0.80])

    panels = [
        (ax1, raw_patches, raw_values, "Before Cleaning (Raw Grid Density)"),
        (ax2, cleaned_patches, cleaned_values, "After Cleaning (Cleaned Grid Density)"),
    ]

    last_collection = None
    for ax, patches, values, title in panels:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.add_image(mapbox, basemap_zoom)

        if len(patches) > 0 and vmin is not None and vmax is not None:
            collection = PatchCollection(
                patches,
                cmap="viridis",
                alpha=0.55,
                linewidths=0.2,
                edgecolor="none",
                zorder=3,
            )
            collection.set_transform(proj)
            collection.set_array(values)
            collection.set_clim(vmin, vmax)
            ax.add_collection(collection)
            last_collection = collection

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}
        ax.set_title(title, fontsize=13, fontweight="bold")

    if last_collection is not None:
        cbar = plt.colorbar(last_collection, cax=cax, orientation="vertical", extend="max")
        cbar.set_label("Low <- Grid Point Count -> High", fontsize=12, fontweight="bold")
        cbar.set_ticks([])
        cbar.ax.tick_params(length=0)

    fig.savefig(save_path, format="png", dpi=300)
    plt.close(fig)


def _write_log(
    log_path: Path,
    figure_path: Path,
    raw_grid_path: Path,
    cleaned_grid_path: Path,
    boundary: tuple[float, float, float, float],
    resolution: float,
    raw_rows: int,
    cleaned_rows: int,
    raw_grid_count: int,
    cleaned_grid_count: int,
) -> None:
    log_text = (
        "Completed before/after grid visualization.\n"
        f"Figure: {figure_path}\n"
        f"Raw grid GeoJSON: {raw_grid_path}\n"
        f"Cleaned grid GeoJSON: {cleaned_grid_path}\n"
        f"Boundary (lon_min, lon_max, lat_min, lat_max): {boundary}\n"
        f"Grid resolution (meters): {resolution}\n"
        f"Raw rows: {raw_rows:,}\n"
        f"Cleaned rows: {cleaned_rows:,}\n"
        f"Raw active grid cells: {raw_grid_count:,}\n"
        f"Cleaned active grid cells: {cleaned_grid_count:,}\n"
    )
    log_path.write_text(log_text, encoding="utf-8")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)

    _bootstrap_local_packages()
    import iogenius as iog

    load_dotenv()
    _set_plot_font()

    root = _project_root()
    iog.set_working_directory(str(root))

    output_dir = Path("./output/P1/p4_points_before_after_plot")
    log_dir = Path("./logs/P1/p4_points_before_after_plot")
    iog.create_new_directory(output_dir)
    iog.create_new_directory(log_dir)

    raw_data_path = Path("./data/bridge_msg_filtered.csv")
    cleaned_feather_path = Path("./output/P1/p2_traj_cleaning/P1_p2_cleaned_traj.feather")
    cleaned_csv_path = Path("./output/P1/p2_traj_cleaning/P1_p2_cleaned_traj.csv")

    df_raw = _load_raw_data(raw_data_path)
    df_cleaned = _load_cleaned_data(cleaned_feather_path, cleaned_csv_path)

    dynamic_extent = _get_dynamic_extent(df_raw=df_raw, df_cleaned=df_cleaned)
    grid_resolution = 10.0

    mapbox_token = os.getenv("MAPBOX_TOKEN")
    if not mapbox_token:
        raise ValueError("Please set MAPBOX_TOKEN in environment or .env")
    mapbox = MapboxTiles(mapbox_token, "light-v10")

    raw_grids = _points_to_grids(df=df_raw, boundary=dynamic_extent, resolution=grid_resolution)
    cleaned_grids = _points_to_grids(df=df_cleaned, boundary=dynamic_extent, resolution=grid_resolution)

    raw_grid_path = output_dir / "P1_p4_raw_grids_res10.geojson"
    cleaned_grid_path = output_dir / "P1_p4_cleaned_grids_res10.geojson"
    raw_grids.to_file(raw_grid_path, driver="GeoJSON")
    cleaned_grids.to_file(cleaned_grid_path, driver="GeoJSON")

    figure_path = output_dir / "P1_p4_grid_before_after_res10.png"
    _plot_before_after_grids(
        raw_grids=raw_grids,
        cleaned_grids=cleaned_grids,
        extent=dynamic_extent,
        save_path=figure_path,
        mapbox=mapbox,
        basemap_zoom=16,
        vmin_percentile=0,
        vmax_percentile=85,
    )

    _write_log(
        log_path=log_dir / "p4_points_before_after_plot.log",
        figure_path=figure_path,
        raw_grid_path=raw_grid_path,
        cleaned_grid_path=cleaned_grid_path,
        boundary=dynamic_extent,
        resolution=grid_resolution,
        raw_rows=len(df_raw),
        cleaned_rows=len(df_cleaned),
        raw_grid_count=len(raw_grids),
        cleaned_grid_count=len(cleaned_grids),
    )
