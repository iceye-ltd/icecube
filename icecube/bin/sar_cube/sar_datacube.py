#!/usr/bin/env python

"""
The script creates SAR datacubes using ICEYE images
"""
import abc
from typing import Tuple, Dict
import xarray as xr
import os
from icecube.utils.common_utils import assert_metadata_exists
import json
from tqdm import tqdm
import pandas as pd
from typing import List
from pathlib import Path

# Local import
from icecube.bin.sar_cube.sar_datacube_metadata import SARDatacubeMetadata
from icecube.utils.common_utils import measure_time, NumpyEncoder
from icecube.bin.config import CubeConfig
from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))

from icecube.bin.datacube_variables import NAME_BAND, CHUNK_SIZE, NAME_INTENSITY_BAND


class SARDatacube:
    """
    Base class for SAR datacubes
    """

    def __init__(self, cube_config: CubeConfig, xarray_datatype="default"):
        self.cube_config = cube_config.get_config_dict()
        self.chunks = CHUNK_SIZE
        self.xrdataset = None
        self.xarray_datatype = xarray_datatype

    @measure_time
    def create(self, product_type: str, raster_dir: str) -> xr.Dataset:
        """
        Create the datacube and the metdata
        """
        metadata_object = SARDatacubeMetadata(self.cube_config)
        metadata_object = metadata_object.compute_metdatadf_from_folder(
            raster_dir, product_type
        )
        # assert_metadata_exists(metadata_object.metadata_df)
        self.set_xarray_dtype(metadata_object.metadata_df)
        ds = self.create_by_metadata(metadata_object)
        self.xrdataset = ds
        return ds

    @measure_time
    def create_from_list(self, list_path: List[Path]) -> xr.Dataset:
        """
        Create the datacube and the metdata
        """
        metadata_object = SARDatacubeMetadata(self.cube_config)
        metadata_object = metadata_object.compute_metdatadf_from_list(list_path)
        ds = self.create_by_metadata(metadata_object)
        self.xrdataset = ds
        return ds

    @classmethod
    @abc.abstractmethod
    def build(self, cube_config: CubeConfig, raster_dir: str):
        raise NotImplementedError("abstract method")

    @classmethod
    @abc.abstractmethod
    def build_from_list(self, cube_config: CubeConfig, list_path: List[Path]):
        raise NotImplementedError("abstract method")

    @classmethod
    @abc.abstractmethod
    def get_master_dtype(self, metadata_df):
        raise NotImplementedError("abstract method")

    @abc.abstractmethod
    def compute_dummy_xrdataset(
        self, metadata: SARDatacubeMetadata
    ) -> Tuple[xr.Dataset, dict]:
        """
            Compute dummy layer
        Args:
            metadata:
        Returns:
            Tuple containing the xarray Dataset and the dict of metadata associated with this layer
        """
        raise NotImplementedError("abstract method")

    @abc.abstractmethod
    def compute_layer_dataset(
        self, raster_path: str, metadata: SARDatacubeMetadata
    ) -> Tuple[xr.Dataset, dict]:
        """
            Compute the datacube layer value
        Args:
            raster_path: path to the raster.
            metadata: Class metadata containing the meta-information of the datacube
        Returns:
            Tuple containing the xarray Dataset and the dict of metadata associated with this layer
        """
        raise NotImplementedError("abstract method")

    @abc.abstractmethod
    def create_by_metadata(self, metadata: SARDatacubeMetadata) -> xr.Dataset:

        sar_xrdataset_seq = []
        list_metadata = []

        # Create a layer for every raster contain inside the folder
        for i, (df_index, df_row) in enumerate(
            tqdm(
                metadata.metadata_df.iterrows(),
                total=metadata.get_lenght(),
                desc="processing rasters for cubes",
            )
        ):

            # We don't have image for this timestamp - we create an empty array to cover this date.
            if pd.isnull(df_row["product_fpath"]):
                raster_xrdataset, cur_metadata = self.compute_dummy_xrdataset(metadata)
            # We do have images - we build an array with GRD/SLC product
            else:
                # Get the full path
                logger.debug(f"Working on {df_row.product_file}")
                raster_fpath = str(df_row["product_fpath"])

                if not os.path.isfile(raster_fpath):
                    raise Exception(
                        f"File located to {raster_fpath} does not exist - "
                        f"Please verify how the SarDatacubeMetadatadf has been built"
                    )

                raster_xrdataset, cur_metadata = self.compute_layer_xrdataset(
                    raster_fpath, metadata
                )
            list_metadata.append(cur_metadata)
            sar_xrdataset_seq.append(raster_xrdataset)

        # Assgin the a new Series that will be used as dimension along which we do the merge.
        metadata.metadata_df[NAME_BAND] = metadata.metadata_df["acquisition_date"]
        concat_xrdataset = xr.concat(
            sar_xrdataset_seq,
            dim=pd.to_datetime(metadata.metadata_df[NAME_BAND]),
            data_vars="all",
            combine_attrs="drop",
        )

        super_dict = self.concat_metadata(list_metadata)
        # Update attrs for each Datavariable within the datacube
        for dv in list(concat_xrdataset.data_vars):
            concat_xrdataset[dv].attrs = super_dict

        return concat_xrdataset

    def concat_metadata(self, list_metadata: list):
        """
        Concatenate metadata as list of keys
        where keys are superset of dict keys from individual product-files
        """
        for cur_metdata in list_metadata:
            try:
                for k, v in cur_metdata.items():
                    pass
            except Exception:
                raise ValueError(f"could not find items for : {cur_metdata}")

        possible_keys = {
            k for cur_metdata in list_metadata for k, v in cur_metdata.items()
        }
        super_dict = {possible_key: [] for possible_key in possible_keys}

        # fill the metada dict.
        for cur_key in possible_keys:
            for cur_metdata in list_metadata:
                # The image metadata contains the specific keyword.
                if cur_key in cur_metdata:
                    # Transform to string as numpy array cannot be saved as netCDF format
                    cur_value = cur_metdata[cur_key]
                    stringified_value = NumpyEncoder.encode(cur_value)
                    super_dict[cur_key].append(stringified_value)
                else:
                    super_dict[cur_key].append("None")

        return super_dict

    def set_xarray_dtype(self, metadata_df):
        """
        If user has passed a specific datatype, keep it otherwise default to master dtype
        """
        supported_datatypes = [
            "uint8",
            "int8",
            "uint16",
            "int16",
            "float32",
            "float64",
        ]

        if self.xarray_datatype == "default":
            self.xarray_datatype = self.get_master_dtype(metadata_df)
        assert (
            self.xarray_datatype in supported_datatypes
        ), f"Supported datatypes for Xarray are: {supported_datatypes}"

    def get_metadata(
        self, attrs_key, data_variable=None, cube_index=None
    ) -> Dict[str, object]:

        # In this case, we work at the dataset level.
        if data_variable is None:
            if attrs_key not in self.xrdataset.attrs:
                raise Exception(
                    f"{attrs_key} not present in the possible keys : {self.xrdataset.attrs.keys()}"
                )

        else:
            if data_variable not in self.xrdataset:
                raise Exception(
                    f"{data_variable} not present in the possible variables : {self.xrdataset.data_vars}"
                )

            else:
                if attrs_key not in self.xrdataset[data_variable].attrs:
                    raise Exception(
                        f"{attrs_key} not present in the possible keys : {self.xrdataset[data_variable].attrs.keys()}"
                    )

                try:
                    return self.xrdataset[data_variable].attrs[attrs_key][cube_index]

                except Exception as e:
                    raise Exception(f"{cube_index} not a valid index")

        if len(self.xrdataset.attrs) > cube_index:
            logger.warning(f"the index :{cube_index} is not present into the datacube")
            return {}

        return json.loads(self.xrdataset.attrs[cube_index])

    def get_dims(self):
        if self.xrdataset is not None:
            return dict(self.xrdataset.dims)
        else:
            return None

    def to_file(self, output_fpath: str, format="netCDF4"):
        if self.xrdataset is None:
            raise Exception("Please reformat the")

        self.xrdataset.to_netcdf(output_fpath, mode="w", format=format)
