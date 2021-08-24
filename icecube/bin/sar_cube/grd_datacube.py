#!/usr/bin/env python

# import
import numpy as np
import xarray as xr
import os
import dask.array as da
import rasterio
from typing import Tuple, List
from pathlib import Path

# Local import
from icecube.utils.common_utils import get_dummy_metadata
from icecube.bin.sar_cube.sar_datacube import SARDatacube
from icecube.bin.config import CubeConfig
from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))

from icecube.bin.datacube_variables import (
    NAME_RANGE,
    NAME_AZIMUTH,
    NAME_INTENSITY_BAND,
    RASTER_DTYPE,
    DEFAULT_FILL_VALUE,
)


def _correct_grd_metadata_key(original_key: str) -> str:
    """
    Change an upper case GRD key to it's SLC metadata equivalent.
    By default this is uppercasing all keys; otherwise if the value in the
    `special_keys` dict will be used.

    Args:
        original_key: input metadata key
        special_keys: dictionary of special keys that require more than just lower case

    Return:
        corrected_key: corrected key name
    """

    special_keys = {
        "POSX": "posX",
        "POSY": "posY",
        "POSZ": "posZ",
        "VELX": "velX",
        "VELY": "velY",
        "VELZ": "velZ",
    }
    if original_key in special_keys:
        corrected_key = special_keys[original_key]

    else:
        corrected_key = original_key.lower()

    return corrected_key


class GRDDatacube(SARDatacube):
    """
    Base class for GRD datacubes
    """

    PRODUCT_TYPE = "GRD"

    def __init__(self, cube_config: CubeConfig, xarray_datatype="default"):
        """
        Initiate the instance SLCDatacube.
        Args:
            cube_config: Class containing the configuration for the datacube
        """
        super().__init__(cube_config)
        self.xarray_datatype = xarray_datatype
        self.fill_value = DEFAULT_FILL_VALUE

    @classmethod
    def build(cls, cube_config: CubeConfig, raster_dir: str):
        grd_datacube = GRDDatacube(cube_config, RASTER_DTYPE)
        ds = grd_datacube.create(cls.PRODUCT_TYPE, raster_dir)
        grd_datacube.xrdataset = ds
        return grd_datacube

    @classmethod
    def build_from_list(
        self, cube_config: CubeConfig, list_path: List[Path]
    ) -> SARDatacube:
        grd_datacube = GRDDatacube(cube_config)
        ds = grd_datacube.create_from_list(list_path)
        grd_datacube.xrdataset = ds
        return grd_datacube

    def compute_dummy_xrdataset(self, metadata) -> Tuple[xr.Dataset, dict]:
        max_shape_azimuth, max_shape_range = metadata.get_master_shape()
        # Create a array fill with nan values
        raster_xdarray = xr.DataArray(
            da.full_like(
                None,
                fill_value=self.fill_value,
                dtype=self.xarray_datatype,
                shape=(max_shape_azimuth, max_shape_range),
                chunks=self.chunks,
            ),
            coords=[
                (NAME_AZIMUTH, np.arange(max_shape_azimuth)),
                (NAME_RANGE, np.arange(max_shape_range)),
            ],
            attrs={},
        )

        return xr.Dataset({NAME_INTENSITY_BAND: raster_xdarray}), get_dummy_metadata()

    def compute_layer_xrdataset(self, raster_path, metadata) -> Tuple[xr.Dataset, dict]:
        i_xr_array, grd_metadata = self._create_xdarray_with_ICEYE_GRD(raster_path)
        return xr.Dataset({NAME_INTENSITY_BAND: i_xr_array}), grd_metadata

    def _create_xdarray_with_ICEYE_GRD(self, grd_fpath) -> xr.DataArray:
        """
        The function can very well be replaced by `xr.open_rasterio(fpath)` but it creates float64 type
        indices for azimuth and range.
        """
        # Open with dask
        rasterio_container = rasterio.open(grd_fpath)

        # use Dask to load the array
        array_dask = da.squeeze(
            xr.open_rasterio(
                grd_fpath, chunks={"band": 1, "x": self.chunks[0], "y": self.chunks[1]}
            )
        )

        grd_metadata = rasterio_container.tags()

        # GRD is uppercase. SLC is lowercase by convention. Reasons are lost in the mists of time
        # To uniform the metadata we convert grd metadata key to lower case
        grd_metadata_corrected = {
            _correct_grd_metadata_key(key): value for key, value in grd_metadata.items()
        }
        az_indx = np.arange(array_dask.shape[0])
        range_indx = np.arange(array_dask.shape[1])

        xdarray = xr.DataArray(
            array_dask,
            coords=[az_indx, range_indx],
            dims=[NAME_AZIMUTH, NAME_RANGE],
            attrs=grd_metadata_corrected,
        )

        return xdarray, grd_metadata_corrected

    def get_master_dtype(self, metadata_df):
        """
        Returns:
            Shape of the master frame, if self.master_id is None, first frame with a shape
        """
        index_master = metadata_df.number_of_azimuth_samples.first_valid_index()
        master_fpath = metadata_df.iloc[index_master]["product_fpath"]

        return str(rasterio.open(master_fpath).read(1).dtype)
