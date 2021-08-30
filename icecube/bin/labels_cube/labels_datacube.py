#!/usr/bin/env python

import os
import json
import pandas as pd
import xarray as xr
import abc
from typing import Tuple
from tqdm import tqdm
from icecube.utils.common_utils import (
    measure_time,
    NumpyEncoder,
    assert_metadata_exists,
)
from icecube.utils.logger import Logger
from icecube.bin.sar_cube.sar_datacube_metadata import SARDatacubeMetadata
from icecube.bin.datacube_variables import NAME_BAND
from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))


class LabelsDatacube:
    """
    Core class for creating labels cube
    """

    def __init__(self):
        super().__init__()
        self.json_labels = None
        self.mask_datatype = None
        self.max_shape_azimuth = None
        self.max_shape_range = None

    @measure_time
    def create(self, product_type: str, labels_fpath: str, raster_dir: str):
        """
        main method of class to create labels cube
        :param product_type: type of product, GRD/SLC
        :param labels_fpath: path/to/file.json containing icecube formatted labels
        :param raster_dir: path/to/dir containing rasters
        """
        metadata_object = SARDatacubeMetadata(self.cube_config)
        metadata_object = metadata_object.compute_metdatadf_from_folder(
            raster_dir, product_type
        )
        assert_metadata_exists(metadata_object.metadata_df)
        self.json_labels = self.read_json(labels_fpath)
        self.mask_datatype = self.get_mask_dtype(metadata_object.metadata_df)
        (
            self.max_shape_azimuth,
            self.max_shape_range,
        ) = metadata_object.get_master_shape()
        self.xrdataset = self.create_by_metadata(metadata_object)
        return self

    def create_by_metadata(self, metadata: SARDatacubeMetadata):
        """
        method to create labels cube using SARDatacubeMetadata object
        :param metadata: SARDatacubeMetadata object
        """
        list_metadata = []
        xdataset_seq = []

        for i, (df_index, df_row) in enumerate(
            tqdm(
                metadata.metadata_df.iterrows(),
                total=metadata.get_lenght(),
                desc="processing rasters for labels cube",
            )
        ):

            # We don't have image for this timestamp - we create an empty array to cover this date.
            if pd.isnull(df_row["product_file"]):
                dummy_xdataset, dummy_metadata = self.compute_dummy_xrdataset()
                xdataset_seq.append(dummy_xdataset)
                list_metadata.append(dummy_metadata)

            # We do have images and we will fetch the relevant labels for that
            else:
                # Get the full path
                logger.debug(f"Working on {df_row.product_file}")

                product_file = str(df_row["product_file"])
                asset_labels = self.get_product_labels_from_json(product_file)

                label_xdataset, label_metadata = self.compute_layer_xrdataset(
                    asset_labels, product_file
                )
                list_metadata.append(label_metadata)
                xdataset_seq.append(label_xdataset)

        ds = xr.concat(
            xdataset_seq, dim=NAME_BAND, data_vars="all", combine_attrs="drop"
        )
        super_dict = self.concat_metadata(list_metadata)

        # Update attrs for each Datavariable within the datacube
        for dv in list(ds.data_vars):
            ds[dv].attrs = super_dict

        return ds

    def concat_metadata(self, list_metadata: list):
        """
        Concatenate metadata as list of keys
        where keys are superset of dict keys from individual product-files
        :param list_metadata: metadata list for each product file to be concatenated in labels cube
        """
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

    def read_json(self, json_fpath):
        with open(json_fpath) as json_file:
            return json.load(json_file)

    @classmethod
    @abc.abstractmethod
    def get_mask_dtype(self, metadata_df):
        raise NotImplementedError("abstract method")

    @abc.abstractmethod
    def compute_dummy_xrdataset(self) -> Tuple[xr.Dataset, dict]:
        """
            Compute dummy layer
        Args:
            None:
        Returns:
            Tuple containing the xarray Dataset and the dict of metadata associated with this layer
        """
        raise NotImplementedError("abstract method")

    @abc.abstractmethod
    def compute_layer_dataset(
        self, asset_labels: dict, product_file: str
    ) -> Tuple[xr.Dataset, dict]:
        """
            Compute the datacube layer value
        Args:
            asset_labels: labels in dict format
            product_file: Name of product file
        Returns:
            Tuple containing the xarray Dataset and the dict of metadata associated with this layer
        """
        raise NotImplementedError("abstract method")

    def get_product_labels_from_json(self, product_file):
        """
        Given a .json object of labels, find the labels of the corresponding raster
        """

        for _, raster_label in enumerate(self.json_labels):
            if raster_label["product_file"] == product_file:
                return raster_label["labels"]

        logger.debug(f"Could not find labels for product_file: {product_file}")
        raise ValueError(f"Could not find the labels for product_file:{product_file}")

    def to_file(self, output_fpath: str, format="netCDF4"):
        """
        save labels cube to output format file
        :param output_fpath: path/to/file.format where labels will be saved.
        """
        if self.xrdataset is None:
            raise Exception("Empty xr.Dataset passed for writing")

        self.xrdataset.to_netcdf(output_fpath, mode="w", format=format)
