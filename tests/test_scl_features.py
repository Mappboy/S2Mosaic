import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from s2mosaic.helpers import SORT_AOI_VALID_DATA, validate_inputs
from s2mosaic.stac_utils import (
    recalculate_top_scl_good_data,
    recalculate_top_scl_metrics,
    sort_items,
)


class _DummySrc:
    def __init__(self, arr):
        self.arr = arr
        self.crs = "EPSG:3857"
        self.transform = from_origin(0, 2, 1, 1)
        self.bounds = (0, 0, 2, 2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, idx):
        return self.arr


def _item(href: str):
    return SimpleNamespace(assets={"SCL": SimpleNamespace(href=href)})


def test_recalculate_top_scl_good_data_deduct_water(monkeypatch):
    scl_1 = np.array([[4, 6], [8, 3]], dtype=np.uint8)
    scl_2 = np.array([[4, 4], [4, 4]], dtype=np.uint8)

    def fake_open(href):
        return _DummySrc(scl_1 if href == "a" else scl_2)

    monkeypatch.setattr("s2mosaic.stac_utils.planetary_computer.sign", lambda href: href)
    monkeypatch.setattr("s2mosaic.stac_utils.rio.open", fake_open)

    df = pd.DataFrame(
        [
            {"item": _item("a"), "orbit": 1, "good_data_pct": 0.0, "datetime": "2024-01-01"},
            {"item": _item("b"), "orbit": 1, "good_data_pct": 7.0, "datetime": "2024-01-02"},
        ]
    )

    updated = recalculate_top_scl_good_data(df, top_n=1, deduct_water=True)

    assert updated.iloc[0]["good_data_pct"] == 25.0
    assert updated.iloc[1]["good_data_pct"] == 7.0


def test_recalculate_top_scl_good_data_land_only(monkeypatch):
    scl = np.array([[4, 6], [8, 3]], dtype=np.uint8)

    monkeypatch.setattr("s2mosaic.stac_utils.planetary_computer.sign", lambda href: href)
    monkeypatch.setattr("s2mosaic.stac_utils.rio.open", lambda href: _DummySrc(scl))
    monkeypatch.setattr(
        "s2mosaic.stac_utils._get_land_mask_for_scene",
        lambda item, shape, cache_dir: np.array([[1, 0], [0, 0]], dtype=bool),
    )

    df = pd.DataFrame(
        [{"item": _item("a"), "orbit": 1, "good_data_pct": 0.0, "datetime": "2024-01-01"}]
    )

    updated = recalculate_top_scl_good_data(df, top_n=1, land_only=True)

    assert updated.iloc[0]["good_data_pct"] == 100.0


def test_recalculate_top_scl_metrics_tracks_aoi_and_land_without_overwriting(monkeypatch):
    scl = np.array([[4, 8], [4, 3]], dtype=np.uint8)
    aoi = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 2)]}, crs="EPSG:3857")

    monkeypatch.setattr("s2mosaic.stac_utils.planetary_computer.sign", lambda href: href)
    monkeypatch.setattr("s2mosaic.stac_utils.rio.open", lambda href: _DummySrc(scl))
    monkeypatch.setattr(
        "s2mosaic.stac_utils._get_land_mask_for_scene",
        lambda item, shape, cache_dir: np.array([[1, 1], [0, 0]], dtype=bool),
    )

    df = pd.DataFrame(
        [
            {
                "item": _item("a"),
                "orbit": 1,
                "good_data_pct": 42.0,
                "datetime": "2024-01-01",
            }
        ]
    )

    updated = recalculate_top_scl_metrics(
        df,
        top_n=1,
        calculate_land=True,
        aoi_polygon_layer=aoi,
    )

    assert updated.iloc[0]["good_data_pct"] == 42.0
    assert updated.iloc[0]["scl_good_data_pct"] == 50.0
    assert updated.iloc[0]["land_good_data_pct"] == 50.0
    assert updated.iloc[0]["aoi_good_data_pct"] == 100.0


def test_sort_items_aoi_valid_data_uses_aoi_land_then_original_good_data():
    df = pd.DataFrame(
        [
            {
                "item": _item("a"),
                "orbit": 1,
                "good_data_pct": 20.0,
                "aoi_good_data_pct": 90.0,
                "land_good_data_pct": 10.0,
                "datetime": "2024-01-01",
            },
            {
                "item": _item("b"),
                "orbit": 1,
                "good_data_pct": 10.0,
                "aoi_good_data_pct": 90.0,
                "land_good_data_pct": 20.0,
                "datetime": "2024-01-02",
            },
            {
                "item": _item("c"),
                "orbit": 2,
                "good_data_pct": 100.0,
                "aoi_good_data_pct": np.nan,
                "land_good_data_pct": np.nan,
                "datetime": "2024-01-03",
            },
        ]
    )

    sorted_items = sort_items(df, SORT_AOI_VALID_DATA)

    assert [row.item.assets["SCL"].href for row in sorted_items.itertuples()] == [
        "b",
        "a",
        "c",
    ]


def test_aoi_valid_data_requires_aoi_polygon_layer():
    with pytest.raises(ValueError, match="aoi_polygon_layer"):
        validate_inputs(
            sort_method=SORT_AOI_VALID_DATA,
            mosaic_method="mean",
            no_data_threshold=0.01,
            required_bands=["B04"],
            grid_id="50HMH",
            percentile_value=None,
            sort_function=None,
        )
