#!/usr/bin/env python

"""
test functions for datacubes with raster labels
...
"""

import os
import shutil
import numpy as np
import rasterio
import json
from pathlib import Path
from icecube.bin.config import CubeConfig
from icecube.bin.labels_cube.labels_cube_generator import LabelsDatacubeGenerator
from icecube.bin.generate_cube import IceyeProcessGenerateCube
from icecube.bin.datacube import Datacube
from icecube.bin.datacube_variables import NAME_LABELS_BAND

res_abspath = os.path.join(Path(__file__).parent, "resources")

grd_raster_dir = os.path.join(res_abspath, "grd_stack")
cube_save_dir = os.path.join(res_abspath, "temp")
masks_raster_dir = os.path.join(res_abspath, "masks")
masks_labels_fpath = os.path.join(res_abspath, "labels/dummy_mask_labels.json")

cube_save_fpath = os.path.join(cube_save_dir, "temp.nc")


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


def delete_temporary_cube_dir(cube_dir):
    shutil.rmtree(cube_dir)


def create_temporary_cube_dir(cube_dir):
    if os.path.exists(cube_dir):
        delete_temporary_cube_dir(cube_dir)

    os.mkdir(cube_dir)


def read_json(json_fpath):
    with open(json_fpath) as f:
        return json.load(f)


def confirm_masks_values_in_cube(cube_save_fpath):
    dc = Datacube().read_cube(cube_save_fpath)
    assert dc.xrdataset[NAME_LABELS_BAND].attrs  # make sure attributes exist
    all_products = dc.get_all_products(dc.get_xrarray(NAME_LABELS_BAND))
    valid_products = [
        product_name for product_name in all_products if product_name != "None"
    ]

    for product_file in valid_products:
        mask_local_fpath = os.path.join(
            masks_raster_dir, product_file.replace(".tif", ".png")
        )
        mask_values = rasterio.open(mask_local_fpath).read(1)

        cube_mask_values = dc.get_product_values(
            product_file, dc.get_xrarray(NAME_LABELS_BAND)
        )
        assert (
            mask_values.all() == cube_mask_values.all()
        ), "mask values should be same in cube as well"

    # Similarly create a check for "None" rasters too.
    invalid_indices = [
        i for i, product_name in enumerate(all_products) if product_name == "None"
    ]
    gt_zeros = np.zeros((10, 10))
    gt_np_nans = np.empty((10, 10))
    gt_np_nans[:] = np.nan

    for i in invalid_indices:
        dummy_values = dc.get_index_values(i, dc.get_xrarray(NAME_LABELS_BAND))

        if str(dummy_values.dtype) == "float32" or str(dummy_values.dtype) == "float64":
            assert dummy_values.all() == gt_np_nans.all()
        else:
            assert dummy_values.all() == gt_zeros.all()


def get_product_labels_from_json(product_file, json_labels):
    for _, raster_label in enumerate(json_labels):
        if raster_label["product_file"] == product_file:
            return raster_label["labels"]

    raise ValueError(f"Could not find the labels for product_file: {product_file}")


def test_grd_masks_labels_default_config():
    """
    Given default configuration of user, create segmentation masks.
    """
    create_run_time_masks_labels()
    product_type = "GRD"

    cc = CubeConfig()
    cc.load_config(None)

    labels_datacube = LabelsDatacubeGenerator.build(
        cc, product_type, masks_labels_fpath, grd_raster_dir
    )

    # test saving the cube and delete then.
    create_temporary_cube_dir(cube_save_dir)
    labels_datacube.to_file(cube_save_fpath)

    confirm_masks_values_in_cube(cube_save_fpath)
    delete_temporary_cube_dir(cube_save_dir)


def test_grd_masks_labels_custom_config():
    """
    Given custom configuration of user, create segmentation masks.
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case4.json")
    create_run_time_masks_labels()
    product_type = "GRD"

    cc = CubeConfig()
    cc.load_config(cube_config_fpath)

    labels_datacube = LabelsDatacubeGenerator.build(
        cc, product_type, masks_labels_fpath, grd_raster_dir
    )

    create_temporary_cube_dir(cube_save_dir)
    labels_datacube.to_file(cube_save_fpath)

    dc = Datacube().read_cube(cube_save_fpath)
    assert (
        len(dc.get_all_products(dc.get_xrarray(NAME_LABELS_BAND))) == 1
    ), "Cannot have more than one images with given configuration"
    confirm_masks_values_in_cube(cube_save_fpath)
    delete_temporary_cube_dir(cube_save_dir)


def test_grd_masks_labels_custom_config2():
    """
    Given custom configuration of user, create segmentation masks.
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case5.json")
    create_run_time_masks_labels()
    product_type = "GRD"

    cc = CubeConfig()
    cc.load_config(cube_config_fpath)

    labels_datacube = LabelsDatacubeGenerator.build(
        cc, product_type, masks_labels_fpath, grd_raster_dir
    )
    create_temporary_cube_dir(cube_save_dir)
    labels_datacube.to_file(cube_save_fpath)

    dc = Datacube().read_cube(cube_save_fpath)
    assert (
        len(dc.get_all_products(dc.get_xrarray(NAME_LABELS_BAND))) == 6
    ), "Must contain 6 products with given configuration"
    confirm_masks_values_in_cube(cube_save_fpath)
    delete_temporary_cube_dir(cube_save_dir)


def test_cube_generator_with_raster_labels():
    """
    test end-end workflow with sample raster labels
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case4.json")

    _ = IceyeProcessGenerateCube.create_cube(
        grd_raster_dir, cube_config_fpath, masks_labels_fpath
    )


def test_mask_dtype():
    """
    Given custom configuration of user, create segmentation masks.
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case4.json")
    product_type = "GRD"

    cc = CubeConfig()
    cc.load_config(cube_config_fpath)

    labels_datacube = LabelsDatacubeGenerator.build(
        cc, product_type, masks_labels_fpath, grd_raster_dir
    )
    assert str(labels_datacube.xrdataset[NAME_LABELS_BAND].dtype) == "uint8"
