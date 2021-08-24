#!/usr/bin/env python

"""
The script provides interface to base classes for generating SAR+Labels datacube
"""

import enum
import os
from pathlib import Path
from icecube.bin.config import CubeConfig
from icecube.bin.sar_cube.sar_datacube import SARDatacube
from icecube.bin.sar_cube.slc_datacube import SLCDatacube
from icecube.bin.sar_cube.grd_datacube import GRDDatacube
from icecube.bin.datacube import Datacube
from icecube.bin.labels_cube.labels_cube_generator import LabelsDatacubeGenerator
from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))


class ProductType(enum.Enum):
    GRD = "GRD"
    SLC = "SLC"


class ProductExtension(enum.Enum):
    GRD = ".tif"
    SLC = ".h5"


class LabelsExtension(enum.Enum):
    exts = ".json"


class IceyeProcessGenerateCube:
    """
    An interface class to SARDatacubes and LabelsDatacubes for easily generating ICEYE datacubes
    """

    @classmethod
    def create_cube(
        self, raster_dir: str, cube_config_path: str, labels_fpath=None
    ) -> Datacube:

        if raster_dir is None or not os.path.isdir(raster_dir):
            logger.error(f"folder {raster_dir} desn't seem to be appropriate")

        cube_config = CubeConfig()
        cube_config.load_config(cube_config_path)

        product_type = None

        # Create SAR datacube
        if all(
            fname.endswith(ProductExtension.GRD.value)
            for fname in os.listdir(raster_dir)
        ):
            product_type = ProductType.GRD.value
            sar_datacube = GRDDatacube.build(cube_config, raster_dir)

        elif all(
            fname.endswith(ProductExtension.SLC.value)
            for fname in os.listdir(raster_dir)
        ):
            product_type = ProductType.SLC.value
            sar_datacube = SLCDatacube.build(cube_config, raster_dir)
        else:
            logger.info(
                f"Some of the files present in the folder {raster_dir} are not .tif or .h5 "
                f"- please make sure to remove the extra file before running the script"
            )
            raise Exception("Cannot proceed due to inconsistent extension naming")

        # Create Labels datacube and merge
        if labels_fpath and labels_fpath.endswith(LabelsExtension.exts.value):
            labels_datacube = LabelsDatacubeGenerator.build(
                cube_config, product_type, labels_fpath, raster_dir
            )

            merged_xdatasets = Datacube.merge_xrdatasets(
                [sar_datacube.xrdataset, labels_datacube.xrdataset]
            )
            return Datacube().set_xrdataset(merged_xdatasets)
        else:
            logger.info(
                f"Skipping labels-cube built, "
                f"either labels-fpath was not provided or inconsistent extension naming found"
            )

        logger.info(f"Datacube {sar_datacube.get_dims()} shape built")
        return Datacube().set_xrdataset(sar_datacube.xrdataset)

    @classmethod
    def create_cube_from_list(
        self, list_path: str, cube_config_path: str
    ) -> SARDatacube:

        # check the input
        if list_path is None or len(list_path) == 0:
            raise Exception(f"impossible to pre-process the {list_path}")

        ext = list_path[0].suffix

        cube_config = CubeConfig()
        cube_config.load_config(cube_config_path)

        if ext == ProductExtension.GRD.value:
            datacube = GRDDatacube.build_from_list(cube_config, list_path)

        elif ext == ProductExtension.SLC.value:
            datacube = SLCDatacube.build_from_list(cube_config, list_path)

        else:
            logger.info(f" the extension of the first file {ext} is not .tif or .h5")
            raise Exception("Cannot proceed due to inconsistent extension naming")

        logger.info(f"Datacube {datacube.get_dims()} shape built")
        return datacube


def sample_labels_workflow():
    import icecube

    resource_dir = os.path.join(
        str(Path(icecube.__file__).parent.parent), "tests/resources"
    )
    grd_raster_dir = os.path.join(resource_dir, "grd_stack")
    masks_labels_fpath = os.path.join(resource_dir, "labels/dummy_mask_labels.json")
    vector_labels_fpath = os.path.join(resource_dir, "labels/dummy_vector_labels.json")

    # cube configuration fpath
    cube_config_fpath = os.path.join(resource_dir, "json_config/config_use_case5.json")
    cube_config_fpath = "/mnt/xor/ICEYE_PACKAGES/icecube/icecube/config.json"
    cube_save_fpath = os.path.join(
        resource_dir, "../../icecube/dataset/test_dataset/test_cube_raster_labels.nc"
    )

    datacube = IceyeProcessGenerateCube.create_cube(
        grd_raster_dir, cube_config_fpath, masks_labels_fpath
    )

    datacube.to_file(cube_save_fpath)


def sample_raster_workflow():
    cur_abspath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    cube_config_path = os.path.join(cur_abspath, "config.json")
    tests_resources = os.path.join(cur_abspath, "../tests/resources")
    raster_dir = os.path.join(tests_resources, "slc_stack")
    save_path = os.path.join(cur_abspath, "dataset", "test_slc_stack1.nc")
    datacube = IceyeProcessGenerateCube.create_cube(raster_dir, cube_config_path)
    datacube.to_file(save_path)


def sample_list_workflow():
    cur_abspath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    cube_config_path = os.path.join(cur_abspath, "config.json")
    list_path = [
        Path(
            os.path.join(
                cur_abspath,
                "..",
                "tests",
                "resources",
                "grd_stack",
                "ICEYE_X9_GRD_SLED_54549_20210427T215124_hollow_10x10pixels_fake_0.tif",
            )
        ),
        Path(
            os.path.join(
                cur_abspath,
                "..",
                "tests",
                "resources",
                "grd_stack",
                "ICEYE_X9_GRD_SLED_54549_20210427T215124_hollow_10x10pixels_fake_1.tif",
            )
        ),
    ]
    save_path = os.path.join(cur_abspath, "dataset", "test_stack1.nc")
    datacube = IceyeProcessGenerateCube.create_cube_from_list(
        list_path, cube_config_path
    )
    datacube.to_file(save_path)


def process_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="CLI support for generating ICEYE datacubes"
    )

    parser.add_argument(
        "raster_dir",
        help="Path/to/directory where raster are stored",
        type=str,
    )
    parser.add_argument(
        "--labels-fpath",
        help="path/to/labels.json (in icecube JSON structure) to populate in datacube (Optional)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cube-save",
        help="path/to/cube.nc where datacube shall be saved (Optional)",
        default=None,
        type=str,
    )
    return parser.parse_args()


def cli():
    args = process_args()

    cur_abspath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    cube_config_fpath = os.path.join(cur_abspath, "config.json")
    datacube = IceyeProcessGenerateCube.create_cube(
        args.raster_dir, cube_config_fpath, labels_fpath=args.labels_fpath
    )

    if args.cube_save is not None:
        datacube.to_file(args.cube_save)


if __name__ == "__main__":
    cli()
