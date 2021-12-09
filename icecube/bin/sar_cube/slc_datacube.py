#!/usr/bin/env python

"""
SAR datacube core class using SLC rasters
"""

import numpy as np
import xarray as xr
import os
import dask.array as da
import h5py
import dask
from typing import Tuple, List
from pathlib import Path

from icecube.bin.sar_cube.sar_datacube import SARDatacube
from icecube.utils.common_utils import get_dummy_metadata
from icecube.bin.config import CubeConfig
from icecube.utils.logger import Logger
from icecube.bin.sar_cube.sar_datacube_metadata import SARDatacubeMetadata

logger = Logger(os.path.basename(__file__))

from icecube.bin.datacube_variables import (
    NAME_RANGE,
    NAME_AZIMUTH,
    NAME_REAL_BAND,
    NAME_COMPLEX_BAND,
    RASTER_DTYPE,
    DEFAULT_FILL_VALUE,
)


class SLCDatacube(SARDatacube):
    """
    Base class for SLC datacubes
    """

    PRODUCT_TYPE = "SLC"

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
    def build(cls, cube_config: CubeConfig, raster_dir: str) -> SARDatacube:
        slc_datacube = SLCDatacube(cube_config, RASTER_DTYPE)
        ds = slc_datacube.create(cls.PRODUCT_TYPE, raster_dir)
        slc_datacube.xrdataset = ds
        return slc_datacube

    @classmethod
    def build_from_list(
        self, cube_config: CubeConfig, list_path: List[Path]
    ) -> SARDatacube:
        slc_datacube = SLCDatacube(cube_config)
        ds = slc_datacube.create_from_list(list_path)
        slc_datacube.xrdataset = ds
        return slc_datacube

    def compute_dummy_xrdataset(
        self, metadata: SARDatacubeMetadata
    ) -> Tuple[xr.Dataset, dict]:
        max_shape_azimuth, max_shape_range = metadata.get_master_shape()
        # Create a array fill with nan values
        raster_xdarray = xr.DataArray(
            # Be careful - if datatype is a int, -1 will become positif
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
        # Create dataset.
        raster_xrdataset = xr.Dataset(
            {
                NAME_REAL_BAND: raster_xdarray,
                NAME_COMPLEX_BAND: raster_xdarray,
                # NAME_AMPLITUDE_BAND: raster_xdarray,
                # NAME_PHASE_BAND: raster_xdarray
            }
        )
        return raster_xrdataset, get_dummy_metadata()

    def compute_layer_xrdataset(
        self, raster_fpath, metadata: SARDatacubeMetadata
    ) -> Tuple[xr.Dataset, dict]:
        max_shape_azimuth, max_shape_range = metadata.get_master_shape()
        s_i_xr_array, s_q_xr_array, slc_metadata = self._create_xdarray_with_ICEYE_SLC(
            raster_fpath, max_shape=(max_shape_azimuth, max_shape_range)
        )

        # Add the raster as layer in the datacube
        raster_xrdataset = xr.Dataset(
            {
                NAME_REAL_BAND: s_i_xr_array,
                NAME_COMPLEX_BAND: s_q_xr_array,
                # NAME_PHASE_BAND: self._compute_phase(s_i_xr_array, s_q_xr_array),
                # NAME_AMPLITUDE_BAND: self._compute_amplitude(s_i_xr_array, s_q_xr_array)
            }
        )

        return raster_xrdataset, slc_metadata

    def _create_xdarray_with_ICEYE_SLC(
        self, slc_fpath, max_shape=None
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        ...
        """
        # Opening Raster
        logger.debug(f"Reading the raster {slc_fpath}")
        hf = h5py.File(slc_fpath, "r")

        # Extracting raster bands
        s_i_h5 = hf["s_i"]
        s_q_h5 = hf["s_q"]

        # use Dask to load the array
        s_i_h5_dask = da.from_array(s_i_h5, chunks=self.chunks)
        s_q_h5_dask = da.from_array(s_q_h5, chunks=self.chunks)

        if s_i_h5.shape != max_shape:
            logger.error(
                f"The shape of the raster is different that the master shape {s_i_h5.shape} != {max_shape}"
            )
            raise Exception(
                "Raster shape are not equivalent - we can't load them into datacube for now."
            )

        # Load the metadata:
        slc_metadata = self._read_slc_metadata(hf)
        az_indx, range_indx = np.arange(s_i_h5.shape[0]), np.arange(s_i_h5.shape[1])

        s_i_array = xr.DataArray(
            s_i_h5_dask,
            coords=[az_indx, range_indx],
            dims=[NAME_AZIMUTH, NAME_RANGE],
            attrs={},
        )

        s_q_array = xr.DataArray(
            s_q_h5_dask,
            coords=[az_indx, range_indx],
            dims=[NAME_AZIMUTH, NAME_RANGE],
            attrs={},
        )

        return s_i_array, s_q_array, slc_metadata

    def _read_slc_metadata(self, h5f):
        """
        Read metadata from SLC header

        Args:
            opened_h5_file: SLC h5 file opened as a h5py File
        """

        meta_dict = {}

        # h5py doesn't read .keys() correctly, but returns a generator. So parsing to list to get all keys.
        key_list = [x for x in h5f.keys()]

        # Check if there already is a list of bands
        if "bands" in key_list:
            non_meta_keys = h5f["bands"]
        else:
            non_meta_keys = ["s_i", "s_q"]

        # Splines are not passed as they are not necessary. We just skip them for now.
        # This doesn't not effect the SLC in any meaningfull way
        additional_bands_to_skip_for_now = [
            "RPC",
            "height_spline",
            "lat_spline",
            "lon_spline",
        ]
        for key in key_list:
            if key not in non_meta_keys + additional_bands_to_skip_for_now:

                h5_meta_val = h5f[key]

                if type(h5_meta_val[()]) == bytes:
                    # h5py 3.0+ introduces str stored as bytes. We use .asstr() to read it (but breaks with h5py < 2.10)
                    meta_dict[key] = np.array_str(np.array(h5_meta_val.asstr()[()]))

                elif type(h5_meta_val[()]) == np.bytes_:
                    # This is most likely a date
                    meta_dict[key] = np.array_str(
                        np.squeeze(np.char.decode(h5_meta_val, "utf-8"))
                    )

                else:
                    # We're handling a numpy array so we should be good to go
                    meta_dict[key] = np.array_str(np.array(h5_meta_val[()]))

                    # RPCs are nested under "RPC/" in the h5 thus need to be parsed in a specific manner
        if "RPC" in h5f:
            RPC_source = h5f["RPC"]

            for key, val in RPC_source.items():
                meta_dict[f"RPC_{key}"] = np.array(val, dtype=np.float32)
        else:
            meta_dict["RPC"] = np.array(None, dtype=np.float32)

        return meta_dict

    def _compute_amplitude(
        self, s_i_array: xr.DataArray, s_q_array: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute the amplitude of the SLC image
        Args:
            s_i_array: real dask array
            s_q_array: complex dask array

        Returns:
            Amplitude as a dask array
        """
        s_i_array_square = s_i_array * s_i_array
        s_q_array_square = s_q_array * s_q_array
        intensity = s_i_array_square + s_q_array_square
        intensity.attrs = s_i_array.attrs
        s_i_array_square = None
        s_q_array_square = None
        return np.sqrt(intensity)

    def _compute_phase(
        self, s_i_array: xr.DataArray, s_q_array: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute the phase of the SLC image
        Args:
            s_i_array: real dask array
            s_q_array: complex dask array

        Returns:
            Phase as a dask array
        """
        phase = da.arctan2(s_q_array, s_i_array)
        phase.attrs = s_i_array.attrs
        return phase

    def get_master_dtype(self, metadata_df):
        """
        Returns:
            Shape of the master frame, if self.master_id is None, first frame with a shape
        """
        index_master = metadata_df.number_of_azimuth_samples.first_valid_index()
        master_fpath = metadata_df.iloc[index_master]["product_fpath"]

        i_band_dtype = str(
            dask.array.from_array(
                h5py.File(master_fpath, "r")["s_i"], chunks=(5000, 5000)
            ).dtype
        )
        q_band_dtype = str(
            dask.array.from_array(
                h5py.File(master_fpath, "r")["s_q"], chunks=(5000, 5000)
            ).dtype
        )

        assert i_band_dtype == q_band_dtype, "Inconsistent datatypes found in bands"
        return i_band_dtype
