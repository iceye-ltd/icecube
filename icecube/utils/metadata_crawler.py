#!/usr/bin/env python
"""
Given directories of ICEYE raster data, crawl and scrape metadata for variables of interest
"""

import os
import warnings
import numpy as np
import pandas as pd
from shapely import geometry
import rasterio
from datetime import datetime
from typing import List
from icecube.utils import analytics_IO as IO
from icecube.utils.common_utils import (
    DirUtils,
)


def metadata_crawler(raster_dir, product_type, variables, recursive=False):
    """
    Go through every ICEYE product in a single folder. Save each field as defined in variables.
    INPUTS:
        folder (str), directory which has the images
        product type (str), either "SLC" or "GRD"
        variables (list, tuple, str), either 'all' or a tuple ('avg_scene_height', 'look_side', ...)
        recursive (bool), whether to go through only the folder or all subfolders too.
    OUTPUTS:
        pandas dataframe with variables as columns and one row per image
    """
    if product_type == "GRD":
        fext = ".tif"
    elif product_type == "SLC":
        fext = ".h5"
    _sanity_check_inputs(raster_dir, product_type, variables, recursive)
    _, raster_paths = DirUtils.get_dir_files(raster_dir, fext=fext)

    return metadata_crawler_list(raster_paths, variables)


def metadata_crawler_list(raster_paths: List[str], variables):
    metadata_dicts = []

    for indx, raster_path in enumerate(raster_paths):
        metadata = IO.load_ICEYE_metadata(raster_path)
        parsed_metadata = _parse_data_row(metadata, variables)
        parsed_metadata["product_fpath"] = raster_path
        
        if pd.isnull(parsed_metadata["product_file"]):
            parsed_metadata["product_file"] = os.path.basename(raster_path)
        
        if pd.isnull(parsed_metadata["number_of_azimuth_samples"]) or pd.isnull(parsed_metadata["number_of_range_samples"]):
            raster_shape = rasterio.open(raster_path).shape
            parsed_metadata["number_of_azimuth_samples"] = raster_shape[0]
            parsed_metadata["number_of_range_samples"] = raster_shape[1]

        metadata_dicts.append(parsed_metadata)

    metadata_df = pd.DataFrame(
        metadata_dicts,
        columns=(variables),
    )
    return metadata_df


def _sanity_check_inputs(folder, product_type, variables, recursive):

    assert isinstance(folder, str)
    assert os.path.exists(folder), "Could not find the folder %s" % folder

    assert isinstance(product_type, str)
    assert product_type in ("SLC", "GRD")

    if isinstance(variables, str):
        assert variables == "all", (
            'Did not understand the input "variables". Expected it to be either a tuple of str or the string "all", but the string "%s" was input.'
            % variables
        )
    else:
        assert isinstance(variables, (list, tuple)), (
            'Did not understand the input "variables". Expected it to be either a tuple of str or the string "all", but a %s datatype variable was input.'
            % type(variables)
        )
        for var in variables:
            assert isinstance(var, str), (
                'Did not understand the input "variables". Expected it to be either a tuple of str or the string "all", but a %s datatype variable was found in the tuple.'
                % type(var)
            )
            assert var in variables, (
                "Could not find the requested variable %s in the supported metadata fields. Please add the variable to image_metadata_crawler.SUPPORTED_VARIABLES and _parse_data_row if you need it."
                % var
            )

    assert isinstance(recursive, bool)


def _list_ICEYE_products_in_folder(folder, product, recursive):

    if not recursive:
        filenames = os.listdir(folder)
    else:
        filenames = _go_through_all_subfolders(folder)

    if product == "SLC":
        suffix = ".h5"
    elif product == "GRD":
        suffix = ".tif"

    product_names = []
    for name in filenames:
        if "ICEYE_" in name and product in name and name.endswith(suffix):

            product_names.append(os.path.join(folder, name + suffix))

    if not product_names:
        warnings.warn(
            "Could not find any ICEYE %s-products with the standard naming scheme in %s."
            % (product, folder),
            RuntimeWarning,
        )
        return None
    else:
        return product_names


def _go_through_all_subfolders(folder):

    filenames = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            filenames.append(os.path.join(root, name))

    return filenames


def _parse_data_row(metadata, variables):
    """
    Check for following metadata keys, If not found, append None instead
    "product_file",
    "incidence_center",
    "look_side",
    "orbit_direction",
    "extent",
    "acquisition_date",
    "acquisition_time",
    "number_of_azimuth_samples",
    "number_of_range_samples"
    """
    metadata_row = {}

    for variable in variables:
        
        if variable == "incidence_center":
            try:
                metadata_row["incidence_center"] = _parse_center_incidence_angle(metadata)
            except:
                warnings.warn("key: {} is missing from the metadata. Appending None.".format(variable.upper()), stacklevel=3)
                metadata_row["incidence_center"] = np.nan
                
        elif variable == "extent":
            try:
                metadata_row["extent"] = get_raster_extent(metadata)
            except:
                warnings.warn("key: {} is missing from the metadata. Appending None.".format(variable.upper()), stacklevel=3)
                metadata_row["extent"] = np.nan        
        
        elif variable == "acquisition_date":
            try:
                acquisition_date, acquisition_time = _parse_acquisition_time(metadata)
                metadata_row["acquisition_date"] = acquisition_date
                metadata_row["acquisition_time"] = acquisition_time
            except:
                warnings.warn("ACQUISITION_DATE/TIME is missing from the metadata.", stacklevel=3)
                metadata_row["acquisition_date"] = np.nan
                metadata_row["acquisition_time"] = np.nan

        elif variable == "acquisition_time":
            pass

        else:
            if variable in metadata:
                metadata_row[variable] = metadata[variable]
            else:
                warnings.warn("key: {} is missing from the metadata. Appending None.".format(variable.upper()), stacklevel=3)
                metadata_row[variable] = np.nan

    return metadata_row


def _parse_acquisition_time(metadata):
    time_obj = datetime.strptime(
        str(metadata["acquisition_end_utc"]), "%Y-%m-%dT%H:%M:%S.%f"
    )
    return datetime.strftime(time_obj, "%Y%m%d"), datetime.strftime(
        time_obj, "%H%M%S.%f"
    )


def _parse_center_incidence_angle(metadata):

    if "local_incidence_angle" in metadata:
        center_incidence_angle = metadata["local_incidence_angle"][
            int(round(len(metadata["local_incidence_angle"]) / 2))
        ]
    elif "incidence_center" in metadata:
        center_incidence_angle = metadata["incidence_center"]
    elif "incidence_near" in metadata and "incidence_far" in metadata:
        center_incidence_angle = (
            metadata["incidence_near"] + metadata["incidence_far"]
        ) / 2
    else:
        center_incidence_angle = None

    return center_incidence_angle


def save_metadata_table_as_excel_table(metadata_table, folder):

    metadata_table.to_excel(os.path.join(folder, "metadata.xlsx"), index=False)


def save_metadata_table_as_csv(metadata_table, folder):

    metadata_table.to_csv(os.path.join(folder, "metadata.csv"), index=False)


def save_image_ids_as_txt(metadata_table, folder):

    image_ids = metadata_table["image_number"]
    num_image_ids = len(image_ids)

    with open(os.path.join(folder, "image_ids.txt"), "w") as path:
        for i in range(num_image_ids):
            if i + 1 < num_image_ids:
                path.write("%s," % image_ids[i])
            else:
                path.write("%s" % image_ids[i])


def get_raster_extent(raster_metadata):
    upper_right = raster_metadata["coord_first_far"][2:]
    bottom_right = raster_metadata["coord_last_far"][2:]
    upper_left = raster_metadata["coord_first_near"][2:]
    bottom_left = raster_metadata["coord_last_near"][2:]
    coords = [
        tuple(bottom_left)[::-1],
        tuple(bottom_right)[::-1],
        tuple(upper_right)[::-1],
        tuple(upper_left)[::-1],
        tuple(bottom_left)[::-1],
    ]
    return geometry.Polygon(coords)
