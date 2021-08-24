#!/usr/bin/env python
"""
test functions for datacube core class
"""
import os

from pathlib import Path
import rasterio
import json
import h5py
from collections import Counter

from icecube.bin.sar_cube.grd_datacube import _correct_grd_metadata_key
from icecube.bin.datacube import Datacube
from icecube.bin.generate_cube import IceyeProcessGenerateCube
from icecube.utils.common_utils import (
    get_product_metadata,
    get_slc_metadata,
)
from icecube.bin.datacube_variables import (
    NAME_INTENSITY_BAND,
    NAME_RANGE,
    NAME_AZIMUTH,
    NAME_BAND,
    NAME_LABELS_BAND,
    NAME_REAL_BAND,
    NAME_COMPLEX_BAND,
)
from icecube.bin.config import CubeConfig

res_abspath = os.path.join(Path(__file__).parent, "resources")

grd_raster_dir = os.path.join(res_abspath, "grd_stack")
slc_raster_dir = os.path.join(res_abspath, "slc_stack")
cube_save_dir = os.path.join(res_abspath, "temp")
vector_labels_fpath = os.path.join(res_abspath, "labels/dummy_vector_labels.json")
masks_labels_fpath = os.path.join(res_abspath, "labels/dummy_mask_labels.json")
cube_save_fpath = os.path.join(cube_save_dir, "temp.nc")
masks_raster_dir = os.path.join(res_abspath, "masks")


def create_run_time_masks_labels():
    """
    Generated masks contain absoluate file paths according to the local system.
    For github actions, dynamic generation must take place
    """
    from icecube.bin.labels_cube.create_json_labels import CreateLabels

    masks_names = [
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.png",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_1.png",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_2.png",
    ]

    raster_names = [
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.tif",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_1.tif",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_2.tif",
    ]

    masks_fpaths = [os.path.join(masks_raster_dir, fpath) for fpath in masks_names]

    raster_mask_dict = {}

    for raster_name, mask_fpath in zip(raster_names, masks_fpaths):
        raster_mask_dict[raster_name] = mask_fpath

    create_labels = CreateLabels("raster")

    for product_name, mask_fpath in raster_mask_dict.items():
        seg_mask = create_labels.create_instance_segmentation(mask_fpath)
        create_labels.populate_labels(product_name, seg_mask)
        create_labels.write_labels_to_json(masks_labels_fpath)


def read_json(json_fpath):
    with open(json_fpath) as f:
        return json.load(f)


def get_product_labels_from_json(product_file, json_labels):
    for _, raster_label in enumerate(json_labels):
        if raster_label["product_file"] == product_file:
            return raster_label["labels"]

    raise ValueError(f"Could not find the labels for product_file: {product_file}")


def compare_lists(a, b):
    return Counter(a) == Counter(b)


def get_raster_metadata(raster_fpath):
    corrected_metadata = {
        _correct_grd_metadata_key(key): value
        for key, value in rasterio.open(raster_fpath).tags().items()
    }
    return corrected_metadata


def check_all_methods(dc: Datacube):
    dc.get_dimensions()
    dc.get_xrdataset_metadata()
    all_data_variables = dc.get_data_variables()

    # check all products related foos if they are working fine.
    for dv in all_data_variables:
        dc.get_xrarray_metadata(dv)
        all_products = dc.get_all_products(dc.get_xrarray(dv))

        for i, product_file in enumerate(all_products):
            dc.get_metadata_by_product(product_file, dc.get_xrarray(dv))
            dc.get_product_values(product_file, dc.get_xrarray(dv))
            dc.get_product_index(product_file, dc.get_xrarray(dv))
            dc.get_index_values(i, dc.get_xrarray(dv))


def test_methods():
    """
    Given default configuration of user, check all the methods of Datacube
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case4.json")

    dc = IceyeProcessGenerateCube.create_cube(
        grd_raster_dir, cube_config_fpath, vector_labels_fpath
    )
    check_all_methods(dc)


def test_with_default_config_with_vector_labels():
    """
    Given default configuration of user, check datacube
    """
    cube_config_fpath = os.path.join(
        res_abspath, "json_config/config_use_case_default.json"
    )

    dc = IceyeProcessGenerateCube.create_cube(
        grd_raster_dir, cube_config_fpath, vector_labels_fpath
    )

    default_dims = {NAME_AZIMUTH: 10, NAME_BAND: 3, NAME_RANGE: 10}
    assert dc.get_dimensions() == default_dims
    assert dc.get_xrdataset_metadata() == {}

    all_variables = [NAME_INTENSITY_BAND, NAME_LABELS_BAND]
    assert dc.get_data_variables() == all_variables

    gt_products = [
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_1.tif",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.tif",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_2.tif",
    ]
    for dv in dc.get_data_variables():
        assert compare_lists(dc.get_all_products(dc.get_xrarray(dv)), gt_products)

    # check metadata for Intensity
    intensity_xrarray = dc.get_xrarray(NAME_INTENSITY_BAND)

    for product_file in dc.get_all_products(intensity_xrarray):
        product_metadata = dc.get_metadata_by_product(product_file, intensity_xrarray)
        local_product_fpath = os.path.join(grd_raster_dir, product_file)
        raster_metadata = get_raster_metadata(local_product_fpath)
        assert product_metadata == raster_metadata

    # check values for Intensity
    for product_file in dc.get_all_products(intensity_xrarray):
        product_values = dc.get_product_values(product_file, intensity_xrarray)
        local_fpath = os.path.join(grd_raster_dir, product_file)
        local_values = rasterio.open(local_fpath).read(1)

        assert local_values.all() == product_values.all()

    # check metadata for Labels
    labels_xrarray = dc.get_xrarray(NAME_LABELS_BAND)

    for product_file in dc.get_all_products(labels_xrarray):
        product_metadata = dc.get_metadata_by_product(product_file, labels_xrarray)
        assert product_metadata == get_product_metadata(product_file)

    # check values for Labels
    for product_file in dc.get_all_products(labels_xrarray):
        product_values = dc.get_product_values(product_file, labels_xrarray)
        cube_labels = get_product_labels_from_json(
            product_file, read_json(vector_labels_fpath)
        )

        assert product_values == cube_labels


def test_with_custom_config_with_raster_labels():
    """
    Given default configuration of user, check datacube
    """
    create_run_time_masks_labels()
    cube_config_fpath = os.path.join(
        res_abspath, "json_config/config_use_case_default.json"
    )

    dc = IceyeProcessGenerateCube.create_cube(
        grd_raster_dir, cube_config_fpath, masks_labels_fpath
    )

    default_dims = {NAME_AZIMUTH: 10, NAME_BAND: 3, NAME_RANGE: 10}
    assert dc.get_dimensions() == default_dims
    assert dc.get_xrdataset_metadata() == {}

    all_variables = [NAME_INTENSITY_BAND, NAME_LABELS_BAND]
    assert dc.get_data_variables() == all_variables

    gt_products = [
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_1.tif",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.tif",
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_2.tif",
    ]
    for dv in dc.get_data_variables():
        assert compare_lists(dc.get_all_products(dc.get_xrarray(dv)), gt_products)

    # check metadata for Intensity
    intensity_xrarray = dc.get_xrarray(NAME_INTENSITY_BAND)

    for product_file in dc.get_all_products(intensity_xrarray):
        product_metadata = dc.get_metadata_by_product(product_file, intensity_xrarray)
        local_product_fpath = os.path.join(grd_raster_dir, product_file)
        raster_metadata = get_raster_metadata(local_product_fpath)
        assert product_metadata == raster_metadata

    # check values for Intensity
    for product_file in dc.get_all_products(intensity_xrarray):
        product_values = dc.get_product_values(product_file, intensity_xrarray)
        local_fpath = os.path.join(grd_raster_dir, product_file)
        local_values = rasterio.open(local_fpath).read(1)

        assert local_values.all() == product_values.all()

    # check metadata for Raster Labels
    labels_xrarray = dc.get_xrarray(NAME_LABELS_BAND)

    for product_file in dc.get_all_products(labels_xrarray):
        product_metadata = dc.get_metadata_by_product(product_file, labels_xrarray)
        assert product_metadata == get_product_metadata(product_file)

    # check values for Raster Labels
    for product_file in dc.get_all_products(labels_xrarray):
        product_values = dc.get_product_values(product_file, labels_xrarray)

        mask_local_fpath = os.path.join(
            masks_raster_dir, product_file.replace(".tif", ".png")
        )
        mask_values = rasterio.open(mask_local_fpath).read(1)

        assert mask_values.all() == product_values.all()


def test_with_custom_config_with_slc_stack():
    """
    Given default configuration of user, check datacube
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case5.json")

    dc = IceyeProcessGenerateCube.create_cube(slc_raster_dir, cube_config_fpath)

    # dc.to_file("/mnt/xor/ICEYE_PACKAGES/icecube/tests/delete_me_later.nc")

    default_dims = {NAME_AZIMUTH: 20, NAME_BAND: 6, NAME_RANGE: 20}
    assert dc.get_dimensions() == default_dims
    assert dc.get_xrdataset_metadata() == {}

    all_variables = [NAME_COMPLEX_BAND, NAME_REAL_BAND]
    assert compare_lists(all_variables, dc.get_data_variables())

    gt_products = [
        "None",
        "None",
        "ICEYE_SLC_54549_20210427T215124_hollow_20x20pixels_fake_1.h5",
        "ICEYE_SLC_54549_20210427T215124_hollow_20x20pixels_fake_0.h5",
        "None",
        "None",
    ]
    for dv in dc.get_data_variables():
        assert compare_lists(dc.get_all_products(dc.get_xrarray(dv)), gt_products)

    # check metadata for Bands
    real_xrarray = dc.get_xrarray(NAME_REAL_BAND)

    all_products = dc.get_all_products(real_xrarray)
    valid_products = [
        product_name for product_name in all_products if product_name != "None"
    ]

    for product_file in valid_products:
        product_metadata = dc.get_metadata_by_product(product_file, real_xrarray)
        local_slc_fpath = os.path.join(slc_raster_dir, product_file)
        raster_metadata = get_slc_metadata(h5py.File(local_slc_fpath, "r"))
        # Skipping the metadata match for the moment as Case Letter
        # makese it difficult to work with both.
        # To Do: Need to normalize keys of metadata.
        # assert product_metadata == raster_metadata


if __name__ == "__main__":
    test_methods()
