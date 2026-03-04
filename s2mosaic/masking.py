from concurrent.futures import ThreadPoolExecutor
import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pystac
import rasterio as rio
import scipy
from omnicloudmask import predict_from_array

from .data_reader import get_full_band


def get_valid_mask(bands: np.ndarray, dilation_count: int = 4) -> np.ndarray:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    # erode mask to remove edge pixels
    if dilation_count > 0:
        no_data = scipy.ndimage.binary_dilation(no_data, iterations=dilation_count)
    return ~no_data


def get_masks(
    item: pystac.Item,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
    debug_cache: Union[Path, None] = None,
    max_dl_workers: int = 4,
    download_scl: bool = False,
    scl_filepath_prefix: Union[str, None] = None,
    return_scl: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    # download RG+NIR bands at 20m resolution for cloud masking
    required_bands = ["B04", "B03", "B8A"]
    download_bands = required_bands + (["SCL"] if download_scl else [])
    get_band_20m = partial(get_full_band, res=20, debug_cache=debug_cache)

    hrefs = [item.assets[band].href for band in download_bands]

    with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
        bands_and_profiles = list(executor.map(get_band_20m, hrefs))


    # Split out SCL if requested
    if download_scl:
        if scl_filepath_prefix is None:
            raise ValueError("scl_filepath must be provided when download_scl=True")
        start_date = datetime.fromisoformat(item.properties['datetime'])

        scl_filepath = Path(scl_filepath_prefix) / f"{start_date.strftime('%Y%m%d')}_SCL.tif"
        scl_band, scl_profile = bands_and_profiles[-1]
        mask_bands_and_profiles = bands_and_profiles[:-1]

        scl_path = Path(scl_filepath)
        scl_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure SCL has a band dimension for rasterio.write
        if scl_band.ndim == 2:
            scl_band = scl_band[np.newaxis, :, :]

        profile = dict(scl_profile)
        profile.setdefault("driver", "GTiff")
        profile["count"] = scl_band.shape[0]

        with rio.open(scl_path.as_posix(), "w", **profile) as dst:
            dst.write(scl_band)
    else:
        mask_bands_and_profiles = bands_and_profiles
        scl_band, scl_profile = None, None

    # Separate bands and profiles
    bands, _ = zip(*mask_bands_and_profiles, strict=False)
    ocm_input = np.vstack(bands)

    mask = (
        predict_from_array(
            input_array=ocm_input,
            batch_size=batch_size,
            inference_dtype=inference_dtype,
        )[0]
        == 0
    )
    # interpolate mask back to 10m
    mask = mask.repeat(2, axis=0).repeat(2, axis=1)
    valid_mask = get_valid_mask(ocm_input)
    valid_mask = valid_mask.repeat(2, axis=0).repeat(2, axis=1)
    if return_scl and scl_band is not None:
        return mask, valid_mask, scl_band.squeeze(), scl_profile
    return mask, valid_mask, None, None
