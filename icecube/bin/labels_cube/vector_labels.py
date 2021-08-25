#!/usr/bin/env python
"""

Description: The script provides functionality to ingest labels into datacube
"""
from typing import Tuple
import xarray as xr
import os

import pickle
from typing import Tuple

# Local import
from icecube.bin.labels_cube.labels_datacube import LabelsDatacube
from icecube.bin.config import CubeConfig
from icecube.utils.common_utils import get_dummy_metadata, get_product_metadata

from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))

from icecube.bin.datacube_variables import NAME_LABELS_BAND


class VectorLabels(LabelsDatacube):
    def __init__(self, cube_config: CubeConfig):
        self.cube_config = cube_config.get_config_dict()
        self.xrdataset = None

    @classmethod
    def build(cls, cube_config, product_type: str, labels_fpath: str, raster_dir: str):
        vector_labelscube = VectorLabels(cube_config)
        vector_labelscube.create(product_type, labels_fpath, raster_dir)

        return vector_labelscube

    def compute_dummy_xrdataset(self) -> Tuple[xr.Dataset, dict]:
        dummy_vector_labels = {"labels": {"objects": []}}
        dummy_labels = pickle.dumps(dummy_vector_labels)
        dummy_xdataset = xr.Dataset({NAME_LABELS_BAND: xr.DataArray(dummy_labels)})

        return dummy_xdataset, get_dummy_metadata()

    def compute_layer_xrdataset(self, asset_labels: dict, product_file: str):
        asset_labels = pickle.dumps(asset_labels)
        label_xdataset = xr.Dataset({NAME_LABELS_BAND: xr.DataArray(asset_labels)})
        return label_xdataset, get_product_metadata(product_file)

    def get_mask_dtype(self, metadata_df):
        """
        No mask needed for vector labels
        """
        return None


def sample_workflow():
    import icecube
    from pathlib import Path

    cube_config_fpath = os.path.join(Path(icecube.__file__).parent, "config.json")
    tests_resources_dir = os.path.join(
        Path(icecube.__file__).parent.parent, "tests/resources/"
    )

    raster_dir = os.path.join(tests_resources_dir, "grd_stack")
    dummy_labels_fpath = os.path.join(
        tests_resources_dir, "labels/dummy_vector_labels.json"
    )

    # create a Cube Config
    cc = CubeConfig()
    cc.load_config(cube_config_fpath)

    product_type = "GRD"  # "SLC"

    VectorLabels.build(cc, product_type, dummy_labels_fpath, raster_dir)


if __name__ == "__main__":
    sample_workflow()
