## S2Mosaic 🛰️🌍

__NOTE: This is a fork of DPIRD-DMA/S2Mosaic with additional scene sorting, SCL, and AOI functionality implemented.__

S2Mosaic is a Python package for creating cloud-free mosaics from Sentinel-2 satellite imagery. It allows users to generate composite images for specified grid areas and time ranges, with various options for scene selection and mosaic creation.

[S2Mosaic blog post here](https://dpird-dma.github.io/blog/S2Mosaic-Creating-Cloud-Free-Sentinel-2-Mosaics/)


## Features 🌟

- Create Sentinel-2 mosaics for specific grid areas and time ranges.
- Flexible scene selection methods: by valid data percentage, AOI valid data percentage, oldest, newest, or a custom sorting function.
- Multiple mosaic creation methods: mean, arbitrary percentile, median or first valid pixel.
- Support for different spectral bands, including visual (RGB) composites.
- State-of-the-art cloud masking using the OmniCloudMask library.
- Export mosaics, scene index masks, SCL mosaics, and water masks as GeoTIFF files, or return the mosaic as a NumPy array.

## Additional Features
- SCL-based re-ranking for the top scenes from `valid_data`, with optional water deduction and land-only scoring.
- AOI scene ranking with `sort_method="aoi_valid_data"` and `aoi_polygon_layer` to prioritise scenes with clear pixels inside the target polygon.
- Land-aware AOI ranking that falls back to land valid data and then original `good_data_pct` when AOI scores are tied or unavailable.
- Custom sorting through `sort_function`, including workflows that manually order scenes around reference dates.
- Optional sorted scene CSV output for auditing scene selection.
- Optional scene index, SCL mosaic, and water mask outputs for QA and downstream analysis.
- Duplicate Sentinel-2 processing baselines are filtered by default so repeated acquisitions keep the latest baseline.
- SCL recalculation helpers can optionally include snow as clear data for vegetation-oriented workflows.

## Changelog 📝
### Version 1.2.11:
    * Added AOI-aware SCL re-ranking, land/water-aware SCL metrics, optional QA outputs, duplicate baseline filtering, and SCL-only output support.
### Version 0.1.9:
    * Added a slight dilation to the no data mask to remove diagonal no data pixels from scene edges.
### version 1.0.0
    * Added support for percentile and median mosaic options and minor optimizations.


## Note 📝

We use OmniCloudMask (OCM)for state-of-the-art cloud and cloud shadow masking. OCM will run significantly faster if an available NVIDIA GPU or MPS accelerator is present.

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/drive/1-vdAAnpzp_VCotTV07cbSC9iQFiD7DcH?usp=sharing 'Try S2Mosaic In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab



## Installation 🛠️

You can install S2Mosaic using pip:
```
pip install s2mosaic
```
Or with uv:
```
uv add s2mosaic
```
## Usage Example 1 🚀

Here's a basic example of how to use S2Mosaic:

```python
from s2mosaic import mosaic
from pathlib import Path

# Create a mosaic for a specific grid area and time range
result = mosaic(
    grid_id="50HMH", # Sentinel-2 scene grid ID
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=2, # Duration to collect data from
    output_dir=Path("output"), # Output directory for mosaic TIFF files
    sort_method="valid_data", # Method to sort potential scenes before download
    mosaic_method="mean", # Approach used to combine scenes
    required_bands=['visual'], # Required Sentinel-2 bands
    no_data_threshold=0.001 # Threshold for early stopping
)

print(f"Mosaic saved to: {result}")
```

This example creates a mosaic for the grid area "50HMH" for the first two months of 2022, using the visual (TCI) product. The scenes are sorted by valid data percentage, and the mosaic is created using the mean of valid pixels. The process stops iterating through scenes once the no_data_threshold is reached.

## Usage Example 2 🔬

Here's another example of how to use S2Mosaic:

```python
from s2mosaic import mosaic

# Create a mosaic for a specific grid area and time range
array, rio_profile = mosaic(
    grid_id="50HMH",
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=2,
    sort_method="valid_data",
    mosaic_method="mean",
    required_bands=["B04", "B03", "B02", "B08"],
    no_data_threshold=0.001
)

print(f"Mosaic array shape: {array.shape}")
```

Similar to the example above but with 16-bit red, green, blue, and NIR bands returned as a NumPy array and rasterio profile.

## Advanced Usage 🧠

S2Mosaic provides several options for customizing the mosaic creation process:

- `sort_method`: Choose between "valid_data", "aoi_valid_data", "oldest", "newest", or "custom" to determine scene selection priority.
- `sort_function`: Provide a callable that receives the scene DataFrame and returns it in the desired order. If provided, it overrides `sort_method`.
- `mosaic_method`: Use "mean" for an average of valid pixels, "median", "percentile" with "percentile_value" for more particular merging, or "first" to use the first valid pixel.
- `required_bands`: Specify which spectral bands to include in the mosaic. Use ["visual"] for an RGB composite.
- `no_data_threshold`: Set the threshold for considering a pixel as no-data. Set to None to process all scenes.
- `ocm_batch_size`: Set the batch size for OmniCloudMask inference (default: 1).
- `ocm_inference_dtype`: Set the data type for OmniCloudMask inference (default: "bf16").
- `additional_query`: Set additional query filters such as {"eo:cloud_cover": {"lt": 80}}
- `ignore_duplicate_items`: Remove duplicate acquisitions and keep the latest Sentinel-2 processing baseline (default: True).
- `recalc_scl_for_top_n`: Recalculate `good_data_pct` for the top candidate scenes from their SCL pixels before mosaicking.
- `top_n_scl_recalc`: Set how many sorted scenes should receive SCL-based metric recalculation.
- `scl_recalc_deduct_water`: Treat SCL water pixels as not-clear during SCL metric recalculation.
- `scl_recalc_land_only`: Recalculate valid data over land only when using SCL re-ranking.
- `aoi_polygon_layer`: Provide a GeoPandas-readable polygon path or GeoDataFrame for `sort_method="aoi_valid_data"`.
- `sorted_items_output`: Write the final sorted scene table to CSV.
- `scene_index_output`: Export a raster showing which sorted scene supplied each output pixel.
- `scl_output`: Export the combined SCL mosaic and water mask alongside the mosaic.
- `scl_only`: Build only the SCL-derived outputs without exporting a spectral mosaic.

For more detailed information on these options and additional functionality, please refer to the function docstring in the source code.

## Performance Tips 🚀
- `ocm_batch_size`: If using a GPU, setting this above the default value (1) will speed up cloud masking. In most cases, a value of 4 works well. If you encounter CUDA errors, try using a lower number.
- `ocm_inference_dtype`: if the device supports it 'bf16' tends to be the fastest option, failing this try 'fp16' then 'fp32'.
- `sort_method`: Using "valid_data" as the sort method tends to be the fastest option if no_data_threshold is not None.
- `mosaic_method`: Using 'first' can be a lot faster than 'mean' as only valid, non cloudy, new pixels are downloaded.

## Contributing 🤝

Contributions to S2Mosaic are welcome! Please feel free to submit pull requests, create issues, or suggest improvements. 🙌

## License 📄

This project is licensed under the MIT License. ⚖️

## Acknowledgments 🙏

This package uses the Planetary Computer STAC API and the OmniCloudMask library for cloud masking.
