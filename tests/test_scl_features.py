import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from types import SimpleNamespace

import numpy as np
import pandas as pd

from s2mosaic.stac_utils import recalculate_top_scl_good_data


class _DummySrc:
    def __init__(self, arr):
        self.arr = arr

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
