import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from tqdm.auto import tqdm

from .data_reader import get_band_with_mask
from .helpers import MOSAIC_FIRST, MOSAIC_MEAN, MOSAIC_PERCENTILE, format_progress
from .masking import get_masks
from .mosaic_utils import calculate_percentile_mosaic

logger = logging.getLogger(__name__)


def download_bands_pool(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    coverage_mask: np.ndarray,
    no_data_threshold: Union[float, None],
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
    debug_cache: Union[Path, None] = None,
    max_dl_workers: int = 4,
    percentile_value: float | None = 50.0,
    mask_output: bool = False,
    download_scl: bool = False,
    scl_prefix_path: str=None,
    scl_only: bool = False,
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any], np.ndarray]]:
    s2_scene_size = 10980
    possible_pixel_count = coverage_mask.sum()

    logger.info(f"Possible pixel count: {possible_pixel_count}")

    if "visual" in required_bands:
        band_count = 3
        band_indexes = [1, 2, 3]
        required_bands = required_bands * 3

    else:
        band_count = len(required_bands)
        band_indexes = [1] * len(required_bands)

    if mosaic_method == MOSAIC_PERCENTILE:
        # For percentile, we need to store all values for each pixel
        all_scene_data = []
    else:
        # For mean and first, use the existing approach
        mosaic = np.zeros((band_count, s2_scene_size, s2_scene_size), dtype=np.float32)

    good_pixel_tracker = np.zeros((s2_scene_size, s2_scene_size), dtype=np.uint16)
    # Track scenes will increase memory size
    if mask_output:
        scene_index_mask = np.zeros((s2_scene_size, s2_scene_size), dtype=np.uint8)

    pbar = tqdm(
        total=len(sorted_scenes),
        desc=format_progress(0, len(sorted_scenes), 100.0),
        leave=False,
        bar_format="{desc}",
    )

    for index, item in enumerate(sorted_scenes["item"].tolist()):
        non_cloud_pixels, valid_pixels = get_masks(
            item=item,
            batch_size=ocm_batch_size,
            inference_dtype=ocm_inference_dtype,
            debug_cache=debug_cache,
            max_dl_workers=max_dl_workers,
            download_scl=download_scl,
            scl_filepath_prefix=scl_prefix_path
        )

        combo_mask = (non_cloud_pixels * valid_pixels).astype(bool)

        if mask_output:
            # Keep track of where each pixel came from in scene order.
            # 0 means no valid source scene was found.
            source_index = min(index + 1, np.iinfo(np.uint8).max)
            first_source_pixels = combo_mask & (scene_index_mask == 0)
            scene_index_mask[first_source_pixels] = source_index
        else:
            scene_index_mask = None
        # if method is first, only download valid,
        # non cloudy pixels that have not been filled,
        # else download all valid non cloudy pixels
        if mosaic_method == MOSAIC_FIRST:
            combo_mask = (good_pixel_tracker == 0) & combo_mask

        good_pixel_tracker += combo_mask

        hrefs_and_indexes = [
            (item.assets[band].href, band_index)
            for band, band_index in zip(required_bands, band_indexes, strict=False)
        ]

        get_band_with_mask_partial = partial(
            get_band_with_mask,
            mask=combo_mask,
            debug_cache_path=debug_cache,
            debug_cache=False,
            mosaic_method=mosaic_method,
        )

        if not scl_only:
            with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
                bands_and_profiles = list(
                    executor.map(get_band_with_mask_partial, hrefs_and_indexes)
                )

            bands = []

            for band, profile in bands_and_profiles:
                bands.append(
                    scipy.ndimage.zoom(
                        band,
                        (s2_scene_size / band.shape[0], s2_scene_size / band.shape[1]),
                        order=0,
                    )
                )
                last_profile = profile

            scene_data = np.array(bands)

            if mosaic_method == MOSAIC_PERCENTILE:
                scene_data = np.where(combo_mask, scene_data, np.nan)
                all_scene_data.append(scene_data)
            else:
                mosaic += scene_data

        completed_of_possible = coverage_mask * (good_pixel_tracker != 0)
        no_data_sum = coverage_mask.sum() - completed_of_possible.sum()
        logger.info(f"No data sum: {no_data_sum}")

        no_data_pct = (1 - (completed_of_possible.sum() / possible_pixel_count)) * 100
        logger.info(f"No data pct: {no_data_pct}")

        pbar.set_description(
            format_progress(index + 1, len(sorted_scenes), no_data_pct)
        )

        if mosaic_method == MOSAIC_FIRST:
            if no_data_sum == 0:
                break

        # if no_data_threshold is set, stop if threshold is reached
        if no_data_threshold is not None:
            if no_data_sum < (possible_pixel_count * no_data_threshold):
                break
        pbar.update(1)

    remaining_scenes = pbar.total - pbar.n
    pbar.update(remaining_scenes)
    pbar.refresh()
    pbar.close()

    if mosaic_method == MOSAIC_PERCENTILE and not scl_only:
        if percentile_value is None:
            raise ValueError("Percentile must be provided for percentile mosaic method")

        max_workers = multiprocessing.cpu_count() // 2

        mosaic = calculate_percentile_mosaic(
            all_scene_data=all_scene_data,
            s2_scene_size=s2_scene_size,
            max_workers=max_workers,
            percentile_value=float(percentile_value),
        )

    if mosaic_method == MOSAIC_MEAN and not scl_only:
        mosaic = np.divide(
            mosaic,
            good_pixel_tracker,
            out=np.zeros_like(mosaic),
            where=good_pixel_tracker != 0,
        )

    if "visual" in required_bands:
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    else:
        mosaic = np.clip(mosaic, 0, 65535).astype(np.int16)

    if scl_only:
        return None, None, None

    return mosaic, last_profile, None
