#!/usr/bin/env python
"""

Description: The script provides functionality to ingest labels into datacube
"""

import os
from typing import Tuple

from icecube.bin.labels_cube.vector_labels import VectorLabels
from icecube.bin.labels_cube.raster_labels import RasterLabels
from icecube.bin.config import CubeConfig
from icecube.bin.labels_cube.labels_utils import get_labels_type

from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))


class LabelsDatacubeGenerator:
    """
    Description here ...
    """

    @classmethod
    def build(cls, cube_config, product_type: str, labels_fpath: str, raster_dir: str):
        labels_type = get_labels_type(labels_fpath)

        if labels_type == "raster":
            labels_cube = RasterLabels.build(
                cube_config, product_type, labels_fpath, raster_dir
            )

        elif labels_type == "vector":
            labels_cube = VectorLabels.build(
                cube_config, product_type, labels_fpath, raster_dir
            )

        else:
            raise ValueError("Could not understood the labels format")

        return labels_cube


def sample_workflow():
    import icecube
    from pathlib import Path

    tests_resources_dir = os.path.join(
        Path(icecube.__file__).parent.parent, "tests/resources/"
    )
    config_dir = os.path.join(tests_resources_dir, "json_config/")
    default_config_fpath = [os.path.join(Path(icecube.__file__).parent, "config.json")]
    custom_config_fpaths = [
        os.path.join(config_dir, p)
        for p in ["config_use_case4.json", "config_use_case5.json"]
    ]
    cube_config_fpaths = default_config_fpath + custom_config_fpaths

    tests_resources_dir = os.path.join(
        Path(icecube.__file__).parent.parent, "tests/resources/"
    )

    raster_dir = os.path.join(tests_resources_dir, "grd_stack")
    dummy_vector_labels_fpath = os.path.join(
        tests_resources_dir, "labels/dummy_vector_labels.json"
    )
    dummy_raster_labels_fpath = os.path.join(
        tests_resources_dir, "labels/dummy_vector_labels.json"
    )

    labels_fpaths = [dummy_vector_labels_fpath, dummy_raster_labels_fpath]
    cc = CubeConfig()
    product_type = "GRD"  # "SLC"

    for labels_fpath in labels_fpaths:
        for config_fpath in cube_config_fpaths:
            cc.load_config(config_fpath)
            labels_datacube = LabelsDatacubeGenerator.build(
                cc, product_type, labels_fpath, raster_dir
            )
            # labels_datacube.to_file("/path/to/my/cube.nc")
