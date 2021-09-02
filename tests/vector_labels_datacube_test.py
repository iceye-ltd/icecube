#!/usr/bin/env python
"""
test functions for datacubes with vector geom as labels
...
"""

import os
import shutil
import json
import pickle
from pathlib import Path
from icecube.bin.config import CubeConfig
from icecube.bin.labels_cube.labels_cube_generator import LabelsDatacubeGenerator
from icecube.bin.generate_cube import IceyeProcessGenerateCube
from icecube.bin.datacube import Datacube
from icecube.bin.datacube_variables import NAME_LABELS_BAND, NAME_BAND

res_abspath = os.path.join(Path(__file__).parent, "resources")

grd_raster_dir = os.path.join(res_abspath, "grd_stack")
cube_save_dir = os.path.join(res_abspath, "temp")
vector_labels_fpath = os.path.join(res_abspath, "labels/dummy_vector_labels.json")

cube_save_fpath = os.path.join(cube_save_dir, "temp.nc")


def delete_temporary_cube_dir(cube_dir):
    shutil.rmtree(cube_dir)


def create_temporary_cube_dir(cube_dir):
    if os.path.exists(cube_dir):
        delete_temporary_cube_dir(cube_dir)

    os.mkdir(cube_dir)


def read_json(json_fpath):
    with open(json_fpath) as f:
        return json.load(f)


def get_product_labels_from_json(product_file, json_labels):
    for _, raster_label in enumerate(json_labels):
        if raster_label["product_file"] == product_file:
            return raster_label["labels"]

    raise ValueError(f"Could not find the labels for product_file: {product_file}")


def confirm_vector_geom_in_cube(cube_save_fpath):
    dc = Datacube().read_cube(cube_save_fpath)
    labels_json = read_json(vector_labels_fpath)

    for indx, geom_dict in enumerate(labels_json):
        json_product_file = geom_dict["product_file"]
        json_object_dict = geom_dict["labels"]
        cube_product_dict = dc.get_product_values(
            json_product_file, dc.get_xrarray(NAME_LABELS_BAND)
        )

        assert json_object_dict == cube_product_dict


def test_grd_vector_labels_default_config():
    """
    Given default configuration of user, create vector masks.
    """
    product_type = "GRD"
    cc = CubeConfig()
    cc.load_config(None)
    labels_datacube = LabelsDatacubeGenerator.build(
        cc, product_type, vector_labels_fpath, grd_raster_dir
    )

    # test saving the cube and delete then.
    create_temporary_cube_dir(cube_save_dir)
    labels_datacube.to_file(cube_save_fpath)

    confirm_vector_geom_in_cube(cube_save_fpath)
    delete_temporary_cube_dir(cube_save_dir)


def test_grd_vector_labels_custom_config():
    """
    Given custom configuration of user, create segmentation masks.
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case5.json")
    product_type = "GRD"

    cc = CubeConfig()
    cc.load_config(cube_config_fpath)

    labels_datacube = LabelsDatacubeGenerator.build(
        cc, product_type, vector_labels_fpath, grd_raster_dir
    )

    create_temporary_cube_dir(cube_save_dir)
    labels_datacube.to_file(cube_save_fpath)

    dc = Datacube().read_cube(cube_save_fpath)
    labels_json = read_json(vector_labels_fpath)

    assert len(dc.xrdataset[NAME_BAND]) == 6

    all_products = dc.get_all_products(dc.get_xrarray(NAME_LABELS_BAND))
    valid_products = [
        product_name for product_name in all_products if product_name != "None"
    ]

    for indx, product_fname in enumerate(valid_products):
        product_json_labels = get_product_labels_from_json(product_fname, labels_json)
        product_cube_labels = dc.get_product_values(
            product_fname, dc.get_xrarray(NAME_LABELS_BAND)
        )
        assert product_json_labels == product_cube_labels

    # Make assertion on one of the NA slice values.
    dummy_cube_dict = pickle.loads(dc.xrdataset["Labels"][0].values)
    dummy_vec_dict = {"labels": {"objects": []}}
    assert dummy_cube_dict == dummy_vec_dict

    delete_temporary_cube_dir(cube_save_dir)


def test_cube_generator_with_vector_labels():
    """
    test end-end workflow with sample vector labels
    """
    cube_config_fpath = os.path.join(res_abspath, "json_config/config_use_case4.json")
    _ = IceyeProcessGenerateCube.create_cube(
        grd_raster_dir, cube_config_fpath, vector_labels_fpath
    )
