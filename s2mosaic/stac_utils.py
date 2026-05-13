import logging
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio as rio
from rasterio import features
from rasterio.warp import transform_bounds
import shapely
from pandas import DataFrame
from pystac import Item
from pystac.item_collection import ItemCollection
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry.polygon import Polygon
from urllib3 import Retry

from .helpers import SORT_AOI_VALID_DATA, SORT_NEWEST, SORT_OLDEST, SORT_VALID_DATA

logger = logging.getLogger(__name__)


SCL_CLEAR_CLASSES = {4, 5, 6, 7}
SCL_SHADOW_CLOUD_CLASSES = {2, 3, 8, 9, 10}
SCL_GOOD_DATA_COLUMN = "scl_good_data_pct"
LAND_GOOD_DATA_COLUMN = "land_good_data_pct"
AOI_GOOD_DATA_COLUMN = "aoi_good_data_pct"


def _get_osm_land_polygons(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "land-polygons-split-4326.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(
            "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip",
            zip_path,
        )
    return zip_path


def _get_land_mask_for_scene(item: Item, shape: tuple[int, int], cache_dir: Path) -> np.ndarray:
    signed_href = planetary_computer.sign(item.assets["SCL"].href)
    with rio.open(signed_href) as src:
        bbox_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        land_zip = _get_osm_land_polygons(cache_dir)
        land = gpd.read_file(
            f"zip://{land_zip}!land_polygons.shp",
            bbox=bbox_wgs84,
        )
        if land.empty:
            return np.zeros(shape, dtype=bool)
        land = land.to_crs(src.crs)
        mask = features.rasterize(
            [(geom, 1) for geom in land.geometry if geom is not None],
            out_shape=shape,
            transform=src.transform,
            fill=0,
            dtype=np.uint8,
        )
    return mask.astype(bool)


def _normalise_cache_dir(cache_dir: Any) -> Path:
    if isinstance(cache_dir, (str, Path)):
        return Path(cache_dir)
    return Path.home() / ".cache" / "s2mosaic"


def _load_aoi_polygon_layer(aoi_polygon_layer: Any) -> gpd.GeoDataFrame:
    if isinstance(aoi_polygon_layer, gpd.GeoDataFrame):
        aoi = aoi_polygon_layer.copy()
    elif isinstance(aoi_polygon_layer, (str, Path)):
        aoi = gpd.read_file(aoi_polygon_layer)
    else:
        raise TypeError(
            "aoi_polygon_layer must be a path readable by GeoPandas or a GeoDataFrame"
        )

    if aoi.empty:
        return aoi
    if aoi.crs is None:
        raise ValueError("aoi_polygon_layer must have a CRS")

    aoi = aoi[aoi.geometry.notna()]
    return aoi[(~aoi.geometry.is_empty) & aoi.geometry.is_valid]


def _rasterize_geometries(
    geometries: Any,
    shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    valid_geometries = [
        geom
        for geom in geometries
        if geom is not None and not geom.is_empty and geom.is_valid
    ]
    if not valid_geometries:
        return np.zeros(shape, dtype=bool)

    mask = features.rasterize(
        [(geom, 1) for geom in valid_geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def _get_aoi_mask_for_scene(
    aoi: gpd.GeoDataFrame,
    src: Any,
    shape: tuple[int, int],
) -> np.ndarray:
    if aoi.empty:
        return np.zeros(shape, dtype=bool)

    aoi_for_scene = aoi.to_crs(src.crs)
    return _rasterize_geometries(aoi_for_scene.geometry, shape, src.transform)


def _get_scl_class_masks(
    scl: np.ndarray,
    deduct_water: bool,
    include_snow: bool,
) -> tuple[np.ndarray, np.ndarray]:
    clear_classes = SCL_CLEAR_CLASSES - ({6} if deduct_water else set())
    if include_snow:
        clear_classes.add(11)

    valid_classes = SCL_CLEAR_CLASSES | SCL_SHADOW_CLOUD_CLASSES
    if include_snow:
        valid_classes.add(11)
    valid_mask = np.isin(scl, list(valid_classes))
    good_mask = np.isin(scl, list(clear_classes))
    return valid_mask, good_mask


def _calculate_good_data_pct(
    valid_mask: np.ndarray,
    good_mask: np.ndarray,
    area_mask: np.ndarray | None = None,
) -> float:
    if area_mask is not None:
        valid_mask = valid_mask & area_mask
        good_mask = good_mask & area_mask

    valid_pixels = np.count_nonzero(valid_mask)
    if valid_pixels == 0:
        return 0.0
    return float(np.count_nonzero(good_mask) / valid_pixels * 100)


def recalculate_top_scl_metrics(
    sorted_items: DataFrame,
    top_n: int = 10,
    deduct_water: bool = False,
    calculate_land: bool = False,
    aoi_polygon_layer: Any = None,
    include_snow: bool = False,
    cache_dir: Any = None,
) -> DataFrame:
    if sorted_items.empty:
        return sorted_items

    updated = sorted_items.copy()
    top_n = min(top_n, len(updated))
    cache_dir = _normalise_cache_dir(cache_dir)
    aoi = (
        _load_aoi_polygon_layer(aoi_polygon_layer)
        if aoi_polygon_layer is not None
        else None
    )

    for column in (
        SCL_GOOD_DATA_COLUMN,
        LAND_GOOD_DATA_COLUMN,
        AOI_GOOD_DATA_COLUMN,
    ):
        if column not in updated.columns:
            updated[column] = np.nan

    for idx in range(top_n):
        item = updated.iloc[idx]["item"]
        signed_href = planetary_computer.sign(item.assets["SCL"].href)
        with rio.open(signed_href) as src:
            scl = src.read(1)
            valid_mask, good_mask = _get_scl_class_masks(
                scl=scl,
                deduct_water=deduct_water,
                include_snow=include_snow,
            )
            updated.at[updated.index[idx], SCL_GOOD_DATA_COLUMN] = (
                _calculate_good_data_pct(valid_mask, good_mask)
            )

            if aoi is not None:
                aoi_mask = _get_aoi_mask_for_scene(aoi, src, scl.shape)
                updated.at[updated.index[idx], AOI_GOOD_DATA_COLUMN] = (
                    _calculate_good_data_pct(valid_mask, good_mask, aoi_mask)
                )

        if calculate_land:
            land_mask = _get_land_mask_for_scene(item, scl.shape, cache_dir)
            updated.at[updated.index[idx], LAND_GOOD_DATA_COLUMN] = (
                _calculate_good_data_pct(valid_mask, good_mask, land_mask)
            )

    return updated


def recalculate_top_scl_good_data(
    sorted_items: DataFrame,
    top_n: int = 10,
    deduct_water: bool = False,
    land_only: bool = False,
    include_snow: bool = False,
    cache_dir=None,
) -> DataFrame:
    if sorted_items.empty:
        return sorted_items

    updated = recalculate_top_scl_metrics(
        sorted_items=sorted_items,
        top_n=top_n,
        deduct_water=deduct_water,
        calculate_land=land_only,
        include_snow=include_snow,
        cache_dir=cache_dir,
    )
    metric_column = LAND_GOOD_DATA_COLUMN if land_only else SCL_GOOD_DATA_COLUMN
    top_n = min(top_n, len(updated))
    for idx in range(top_n):
        metric_value = updated.iloc[idx][metric_column]
        if not pd.isna(metric_value):
            updated.at[updated.index[idx], "good_data_pct"] = metric_value

    return updated


def add_item_info(items: ItemCollection) -> DataFrame:
    """Split items by orbit and sort by no_data"""

    items_list = []
    for item in items:
        nodata = item.properties["s2:nodata_pixel_percentage"]
        data_pct = 100 - nodata

        cloud = item.properties["s2:high_proba_clouds_percentage"]
        shadow = item.properties["s2:cloud_shadow_percentage"]
        good_data_pct = data_pct * (1 - (cloud + shadow) / 100)
        capture_date = item.datetime

        items_list.append(
            {
                "item": item,
                "orbit": item.properties["sat:relative_orbit"],
                "good_data_pct": good_data_pct,
                "datetime": capture_date,
            }
        )

    items_df = pd.DataFrame(items_list)
    return items_df


def search_for_items(
    bounds: Polygon,
    grid_id: str,
    start_date: date,
    end_date: date,
    additional_query: Dict[str, Any],
    ignore_duplicate_items: bool = True,
) -> ItemCollection:
    base_query = {"s2:mgrs_tile": {"eq": grid_id}}
    if additional_query:
        base_query.update(additional_query)

    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": shapely.to_geojson(bounds),
        "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        "query": base_query,
    }

    logger.info(
        f"""Searching for items in grid {grid_id} from 
        {start_date} to {end_date} with query: {query}"""
    )

    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=None,
    )
    stac_api_io = StacApiIO(max_retries=retry)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", stac_io=stac_api_io
    )
    items = catalog.search(**query).item_collection()
    logger.info(f"Found {len(items)}")

    if ignore_duplicate_items:
        items = filter_latest_processing_baselines(items)
        logger.info(f"After filtering, {len(items)} items remain")

    return items


def sort_items(items: DataFrame, sort_method: str) -> DataFrame:
    # Sort the dataframe by selected method then by orbit
    if sort_method == SORT_VALID_DATA:
        items_sorted = items.sort_values("good_data_pct", ascending=False)
        items_sorted = _interleave_orbits(items_sorted)
    elif sort_method == SORT_AOI_VALID_DATA:
        sort_columns = [
            AOI_GOOD_DATA_COLUMN,
            LAND_GOOD_DATA_COLUMN,
            "good_data_pct",
        ]
        items_sorted = items.copy()
        for column in sort_columns:
            if column not in items_sorted.columns:
                items_sorted[column] = np.nan
        items_sorted = items_sorted.sort_values(
            sort_columns,
            ascending=[False, False, False],
            na_position="last",
        )
        recalculated_items = _interleave_orbits(
            items_sorted[items_sorted[AOI_GOOD_DATA_COLUMN].notna()]
        )
        unrecalculated_items = _interleave_orbits(
            items_sorted[items_sorted[AOI_GOOD_DATA_COLUMN].isna()]
        )
        items_sorted = pd.concat(
            [recalculated_items, unrecalculated_items],
            ignore_index=True,
        )
    elif sort_method == SORT_OLDEST:
        items_sorted = items.sort_values("datetime", ascending=True).reset_index(
            drop=True
        )
    elif sort_method == SORT_NEWEST:
        items_sorted = items.sort_values("datetime", ascending=False).reset_index(
            drop=True
        )
    else:
        raise Exception(
            "Invalid sort method, must be valid_data, aoi_valid_data, oldest or newest"
        )

    return items_sorted


def _interleave_orbits(items_sorted: DataFrame) -> DataFrame:
    if items_sorted.empty:
        return items_sorted.reset_index(drop=True)

    orbits = items_sorted["orbit"].unique()
    orbit_groups = {
        orbit: items_sorted[items_sorted["orbit"] == orbit] for orbit in orbits
    }

    result = []

    while any(len(group) > 0 for group in orbit_groups.values()):
        for orbit in orbits:
            if len(orbit_groups[orbit]) > 0:
                result.append(orbit_groups[orbit].iloc[0])
                orbit_groups[orbit] = orbit_groups[orbit].iloc[1:]

    return pd.DataFrame(result).reset_index(drop=True)


def filter_latest_processing_baselines(
    items: ItemCollection,
) -> ItemCollection:
    """
    Filter STAC items to keep only the latest processing
    baseline for each unique acquisition.
    """
    if len(items) == 0:
        return items

    # Group items by acquisition (same datetime + tile)
    acquisition_groups: Dict[str, List[Dict[str, Any]]] = {}

    for item in items:
        # Create unique key for this acquisition
        datetime_str: str = (
            item.datetime.strftime("%Y%m%dT%H%M%S") if item.datetime else "unknown"
        )
        tile_id: str = item.properties.get("s2:mgrs_tile", "unknown")
        acquisition_key: str = f"{datetime_str}_{tile_id}"

        # Get processing baseline from properties
        baseline_str: str = item.properties.get("s2:processing_baseline", "0.00")
        # Convert to number for comparison (e.g., '05.11' -> 5.11)
        baseline_num: float = float(baseline_str)

        if acquisition_key not in acquisition_groups:
            acquisition_groups[acquisition_key] = []

        acquisition_groups[acquisition_key].append(
            {"item": item, "baseline": baseline_str, "baseline_num": baseline_num}
        )

    # Keep only the latest baseline for each acquisition
    filtered_items: List[Item] = []
    for acquisition_key, group in acquisition_groups.items():
        if len(group) == 1:
            # No duplicates
            filtered_items.append(group[0]["item"])
        else:
            # Keep the highest baseline number
            latest = max(group, key=lambda x: x["baseline_num"])
            filtered_items.append(latest["item"])
            logger.info(
                f"Filtered {acquisition_key}: kept {latest['baseline']}, "
                f"removed {[x['baseline'] for x in group if x != latest]}"
            )

    return ItemCollection(filtered_items)
