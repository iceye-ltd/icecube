#!/usr/bin/env python
"""

Description: The script provides functionality to ingest labels into datacube for segemntation workflow
"""
import xarray as xr
import os
import numpy as np
import dask
import rasterio

from icecube.bin.labels_cube.labels_datacube import LabelsDatacube
from icecube.bin.config import CubeConfig
from icecube.utils.common_utils import get_dummy_metadata, get_product_metadata
from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))

from icecube.bin.datacube_variables import (
    NAME_LABELS_BAND,
    NAME_RANGE,
    NAME_AZIMUTH,
    CHUNK_SIZE,
    DEFAULT_FILL_VALUE,
    MASK_DTYPE,
)


class RasterLabels(LabelsDatacube):
    def __init__(self, cube_config: CubeConfig, mask_datatype="default"):
        self.cube_config = cube_config.get_config_dict()
        self.xrdataset = None
        self.chunks = CHUNK_SIZE
        self.mask_datatype = mask_datatype
        self.fill_value = DEFAULT_FILL_VALUE

    @classmethod
    def build(cls, cube_config, product_type: str, labels_fpath: str, raster_dir: str):
        raster_labelscube = RasterLabels(cube_config, MASK_DTYPE)
        raster_labelscube.create(product_type, labels_fpath, raster_dir)

        return raster_labelscube

    def compute_dummy_xrdataset(self):
        self._assert_valid_shape()

        # Create a array fill with nan values
        dummy_xda = xr.DataArray(
            dask.array.full_like(
                None,
                fill_value=self.fill_value,
                dtype=self.mask_datatype,
                shape=(self.max_shape_azimuth, self.max_shape_range),
                chunks=self.chunks,
            ),
            coords=[
                (NAME_AZIMUTH, np.arange(self.max_shape_azimuth)),
                (NAME_RANGE, np.arange(self.max_shape_range)),
            ],
            attrs={},
        )

        return xr.Dataset({NAME_LABELS_BAND: dummy_xda}), get_dummy_metadata()

    def _assert_valid_shape(self):
        assert (
            self.max_shape_azimuth != None
        ), "Couldn't find shape of the image in Azimuth"
        assert (
            self.max_shape_range != None
        ), "Couldn't find shape of the image in Azimuth"

    def compute_layer_xrdataset(self, asset_labels: dict, product_file: str):
        label_xdataset = xr.Dataset(
            {
                NAME_LABELS_BAND: self._create_xdarray_with_mask(
                    asset_labels["segmentation"]
                )
            }
        )
        return label_xdataset, get_product_metadata(product_file)

    def _create_xdarray_with_mask(self, mask_fpath) -> xr.DataArray:
        array_dask = dask.array.squeeze(
            dask.array.from_array(
                rasterio.open(mask_fpath).read(1),
                chunks=(self.chunks[0], self.chunks[1]),
            )
        )
        x_axes = np.arange(array_dask.shape[0])
        y_axes = np.arange(array_dask.shape[1])

        return xr.DataArray(
            array_dask,
            coords=[x_axes, y_axes],
            dims=[NAME_AZIMUTH, NAME_RANGE],
            attrs={},
        )

    def get_mask_fpath(self, product_file):
        asset_labels = self.get_product_labels_from_json(product_file)
        return str(asset_labels["segmentation"])

    def get_mask_dtype(self, metadata_df):
        index_master = metadata_df.number_of_azimuth_samples.first_valid_index()
        master_fproduct = metadata_df.iloc[index_master]["product_file"]
        mask_fpath = self.get_mask_fpath(master_fproduct)
        return str(rasterio.open(mask_fpath).read(1).dtype)

    def get_mask_shape(self, metadata_df):
        index_master = metadata_df.number_of_azimuth_samples.first_valid_index()
        master_fproduct = metadata_df.iloc[index_master]["product_file"]
        mask_fpath = self.get_mask_fpath(master_fproduct)
        return str(rasterio.open(mask_fpath).read(1).dtype)


def sample_workflow():
    import icecube
    from pathlib import Path

    tests_resources_dir = os.path.join(
        Path(icecube.__file__).parent.parent, "tests/resources/"
    )

    default_config_fpath = os.path.join(Path(icecube.__file__).parent, "config.json")
    custom_config_fpath = os.path.join(
        tests_resources_dir, "json_config/config_use_case5.json"
    )
    config_fpaths = [default_config_fpath, custom_config_fpath]
    raster_dir = os.path.join(tests_resources_dir, "grd_stack")
    dummy_labels_fpath = os.path.join(
        tests_resources_dir, "labels/dummy_mask_labels.json"
    )

    for index, config_fpath in enumerate(config_fpaths):
        print(
            "\n\n Working on config-file: {} \n\n".format(
                os.path.basename(config_fpath)
            )
        )
        # create a Cube Config
        cc = CubeConfig()
        cc.load_config(config_fpath)

        product_type = "GRD"  # "SLC"

        raster_cube = RasterLabels.build(
            cc, product_type, dummy_labels_fpath, raster_dir
        )
        # raster_cube.to_file("/path/to/ice/cube.nc")
