import os
import tempfile
from pathlib import Path
import rasterio
import numpy as np

# Local import
from icecube.bin.sar_cube.grd_datacube import GRDDatacube
from icecube.bin.datacube_variables import (
    NAME_INTENSITY_BAND,
    NAME_RANGE,
    NAME_AZIMUTH,
    NAME_BAND,
)
from icecube.bin.config import CubeConfig

GRD_RASTER_DIR = os.path.join(Path(__file__).parent, "resources", "grd_stack")
AZIMUTH_DIM = RANGE_DIM = 10
NUMBER_BANDS = 3
START_DATE = 20210426
END_DATE = 20210528

LIST_ORDER_BY_ACQUISITION_DATE = [
    "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_1.tif",
    "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.tif",
    "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_2.tif",
]

EXPECTED_METADATA_VALUES = {
    "acquisition_end_utc": [
        "2021-04-27T21:51:30.025535",
        "2021-04-28T21:51:30.025535",
        "2021-05-27T21:51:30.025535",
    ],
    "orbit_direction": ["DESCENDING", "ASCENDING", "ASCENDING"],
    "incidence_center": ["29.5", "30.5", "28.5"],
    "satellite_look_angle": ["29", "30", "28"],
    "product_file": LIST_ORDER_BY_ACQUISITION_DATE,
}


def _get_metatdata_keys(datacube):
    return datacube.xrdataset[NAME_INTENSITY_BAND].attrs.keys()


def test_grd_datacube():

    # Load the default config
    cube_config = CubeConfig()
    cube_config.load_config(None)

    datacube = GRDDatacube.build(cube_config, GRD_RASTER_DIR)
    # Verify the dimension.
    final_shape = dict(datacube.xrdataset.dims)
    assert final_shape[NAME_AZIMUTH] == AZIMUTH_DIM
    assert final_shape[NAME_RANGE] == RANGE_DIM
    assert final_shape[NAME_BAND] == NUMBER_BANDS

    # verify the content itself
    for i, name_file in enumerate(LIST_ORDER_BY_ACQUISITION_DATE):
        numpy_array_to_compare = rasterio.open(
            os.path.join(GRD_RASTER_DIR, name_file)
        ).read(1)
        assert np.array_equal(
            datacube.xrdataset[NAME_INTENSITY_BAND][i].values, numpy_array_to_compare
        )

    # check the metadata.
    for key, items in EXPECTED_METADATA_VALUES.items():
        for i, item in enumerate(items):
            assert datacube.get_metadata(key, NAME_INTENSITY_BAND, i) == item


def test_save_grd_datacube():

    # Load the default config
    cube_config = CubeConfig()
    cube_config.load_config(None)

    datacube = GRDDatacube.build(cube_config, GRD_RASTER_DIR)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_name = os.path.join(temp_dir, "test_grd_stack.nc")
        datacube.to_file(temp_file_name)
        assert os.path.isfile(temp_file_name)


def test_dummy_data():

    cube_config = CubeConfig()
    cube_config.temporal_resolution = 1
    cube_config.start_date = START_DATE
    cube_config.end_date = END_DATE
    cube_config.load_config(None)

    datacube = GRDDatacube.build(cube_config, GRD_RASTER_DIR)

    # Verify the dimension.
    final_shape = dict(datacube.xrdataset.dims)
    assert final_shape[NAME_AZIMUTH] == AZIMUTH_DIM
    assert final_shape[NAME_RANGE] == RANGE_DIM
    # 33 days between the begining and the end
    assert final_shape[NAME_BAND] == 33

    # verify the content itself - first by checking the nan values
    # value = _get_np_nan()

    for i in range(33):
        if i not in [1, 2, 31]:
            if (
                datacube.xarray_datatype == "float32"
                or datacube.xarray_datatype == "float64"
            ):
                assert np.isnan(
                    datacube.xrdataset[NAME_INTENSITY_BAND][i].values[()]
                ).all()
            else:
                assert np.where(
                    datacube.xrdataset[NAME_INTENSITY_BAND][i].values[()] == 0,
                    True,
                    False,
                ).all()

            for key in _get_metatdata_keys(datacube):
                assert datacube.get_metadata(key, NAME_INTENSITY_BAND, i) == "None"

    # verify the content itself
    intersting_id_in_cubes = [1, 2, 31]
    for i, name_file in enumerate(LIST_ORDER_BY_ACQUISITION_DATE):
        numpy_array_to_compare = rasterio.open(
            os.path.join(GRD_RASTER_DIR, name_file)
        ).read(1)
        index_cube = intersting_id_in_cubes[i]
        assert np.array_equal(
            datacube.xrdataset[NAME_INTENSITY_BAND][index_cube].values,
            numpy_array_to_compare,
        )

    for key, items in EXPECTED_METADATA_VALUES.items():
        for i, item in enumerate(items):
            index_cube = intersting_id_in_cubes[i]
            assert datacube.get_metadata(key, NAME_INTENSITY_BAND, index_cube) == item


def test_grd_datacube_from_list():

    # Load the default config
    cube_config = CubeConfig()
    cube_config.load_config(None)

    list_path = [
        Path(os.path.join(GRD_RASTER_DIR, cur_path))
        for cur_path in [
            "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_0.tif",
            "ICEYE_GRD_54549_20210427T215124_hollow_10x10pixels_fake_1.tif",
        ]
    ]
    datacube = GRDDatacube.build_from_list(cube_config, list_path)

    # Verify the dimension.
    final_shape = dict(datacube.xrdataset.dims)
    assert final_shape[NAME_AZIMUTH] == AZIMUTH_DIM
    assert final_shape[NAME_RANGE] == RANGE_DIM
    assert final_shape[NAME_BAND] == len(list_path)

    # verify the content itself
    for i, name_file in enumerate(LIST_ORDER_BY_ACQUISITION_DATE[:2]):
        numpy_array_to_compare = rasterio.open(
            os.path.join(GRD_RASTER_DIR, name_file)
        ).read(1)
        print(numpy_array_to_compare)
        print(name_file)
        print(datacube.xrdataset[NAME_INTENSITY_BAND][i].values)
        assert np.array_equal(
            datacube.xrdataset[NAME_INTENSITY_BAND][i].values, numpy_array_to_compare
        )

    # verify the content itself
    for i, name_file in enumerate(LIST_ORDER_BY_ACQUISITION_DATE[:2]):
        numpy_array_to_compare = rasterio.open(
            os.path.join(GRD_RASTER_DIR, name_file)
        ).read(1)
        assert np.array_equal(
            datacube.xrdataset[NAME_INTENSITY_BAND][i].values, numpy_array_to_compare
        )

    # check the metadata.
    for key, items in EXPECTED_METADATA_VALUES.items():
        for i, item in enumerate(items[:2]):
            assert datacube.get_metadata(key, NAME_INTENSITY_BAND, i) == item
