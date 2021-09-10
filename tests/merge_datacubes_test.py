#!/usr/bin/env python
"""
The script tests the merging of SARCubes and LabelsCube
"""

import os
import shutil
from pathlib import Path
from icecube.bin.generate_cube import IceyeProcessGenerateCube

 
res_abspath = os.path.join(Path(__file__).parent, "resources")

grd_raster_dir = os.path.join(res_abspath, "grd_stack")
cube_save_dir = os.path.join(res_abspath, "temp")
masks_raster_dir = os.path.join(res_abspath, "masks")
masks_labels_fpath = os.path.join(res_abspath, "labels/dummy_mask_labels.json")
vector_labels_fpath = os.path.join(res_abspath, "labels/dummy_vector_labels.json")

cube_save_fpath = os.path.join(cube_save_dir, "temp.nc")


# We will create a copy of a raster to test the pipeline for same metadata

raster_to_copy = os.path.join(grd_raster_dir, "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.tif")
copied_raster = os.path.join(grd_raster_dir, "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0_copy.tif") 
    
shutil.copy(raster_to_copy, copied_raster)


def delete_temporary_cube_dir(cube_dir):
    shutil.rmtree(cube_dir)


def create_temporary_cube_dir(cube_dir):
    if os.path.exists(cube_dir):
        delete_temporary_cube_dir(cube_dir)

    os.mkdir(cube_dir)


def create_run_time_masks_labels():
    """
    Generated masks contain absoluate file paths according to the local system.
    For github actions, dynamic generation must take place
    """
    from icecube.bin.labels_cube.create_json_labels import CreateLabels

    masks_names = [
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.png"
        ]

    raster_names = [
        "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.tif"
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


def test_xrdatasets_merge_for_raster_labels():
    create_run_time_masks_labels()
    
    def_cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case_default.json")
    custom_cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case6.json") 
    cube_configs_seq = [def_cube_config_fpath, custom_cube_config_fpath]

    create_temporary_cube_dir(cube_save_dir)
    
    for cube_config_fpath in cube_configs_seq:
        datacube = IceyeProcessGenerateCube.create_cube(
            grd_raster_dir, cube_config_fpath, masks_labels_fpath
        )
        
        datacube.to_file(cube_save_fpath)

    delete_temporary_cube_dir(cube_save_dir)


def test_xrdatasets_merge_for_vector_labels():
    create_run_time_masks_labels()

    def_cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case_default.json")
    custom_cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case6.json") 
    cube_configs_seq = [def_cube_config_fpath, custom_cube_config_fpath]

    create_temporary_cube_dir(cube_save_dir)
    
    for cube_config_fpath in cube_configs_seq:
        datacube = IceyeProcessGenerateCube.create_cube(
            grd_raster_dir, cube_config_fpath, vector_labels_fpath
        )
        
        datacube.to_file(cube_save_fpath)

    delete_temporary_cube_dir(cube_save_dir)
    os.remove(copied_raster)