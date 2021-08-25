import os
import tempfile
from pathlib import Path
import h5py
import numpy as np

# Local import
from icecube.bin.sar_cube.slc_datacube import SLCDatacube
from icecube.bin.datacube_variables import (
    NAME_REAL_BAND,
    NAME_COMPLEX_BAND,
    NAME_RANGE,
    NAME_AZIMUTH,
    NAME_BAND,
)
from icecube.bin.config import CubeConfig

SLC_RASTER_DIR = os.path.join(Path(__file__).parent, "resources", "slc_stack")
START_DATE = 20210426
END_DATE = 20210528
AZIMUTH_DIM = RANGE_DIM = 20
NUMBER_BANDS = 3

LIST_ORDER_BY_ACQUISITION_DATE = [
    "ICEYE_SLC_54549_20210427T215124_hollow_20x20pixels_fake_1.h5",
    "ICEYE_SLC_54549_20210427T215124_hollow_20x20pixels_fake_0.h5",
    "ICEYE_SLC_54549_20210427T215124_hollow_20x20pixels_fake_2.h5",
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
    return datacube.xrdataset[NAME_REAL_BAND].attrs.keys()


def test_slc_datacube():

    # Load the default config
    cube_config = CubeConfig()
    cube_config.load_config(None)

    datacube = SLCDatacube.build(cube_config, SLC_RASTER_DIR)

    # Verify the dimension.
    final_shape = dict(datacube.xrdataset.dims)
    assert final_shape[NAME_AZIMUTH] == AZIMUTH_DIM
    assert final_shape[NAME_RANGE] == AZIMUTH_DIM
    assert final_shape[NAME_BAND] == NUMBER_BANDS

    # verify the content itself
    for i, name_file in enumerate(LIST_ORDER_BY_ACQUISITION_DATE):
        hf = h5py.File(os.path.join(SLC_RASTER_DIR, name_file), "r")
        # Extracting raster bands
        s_i_h5 = np.array(hf["s_i"])
        s_q_h5 = np.array(hf["s_q"])

        assert np.array_equal(datacube.xrdataset[NAME_REAL_BAND][i].values, s_i_h5)
        assert np.array_equal(datacube.xrdataset[NAME_COMPLEX_BAND][i].values, s_q_h5)

    # check the metadata.
    for key, items in EXPECTED_METADATA_VALUES.items():
        for i, item in enumerate(items):
            assert datacube.get_metadata(key, NAME_REAL_BAND, i) == item


def test_save_slc_datacube():

    # Load the default config
    cube_config = CubeConfig()
    cube_config.load_config(None)

    datacube = SLCDatacube.build(cube_config, SLC_RASTER_DIR)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_name = os.path.join(temp_dir, "test_slc_stack.nc")
        datacube.to_file(temp_file_name)
        assert os.path.isfile(temp_file_name)


def test_dummy_data_creation():

    cube_config = CubeConfig()
    # Test 1D between frame
    cube_config.temporal_resolution = 1
    cube_config.start_date = START_DATE
    cube_config.end_date = END_DATE
    cube_config.load_config(None)

    datacube = SLCDatacube.build(cube_config, SLC_RASTER_DIR)

    # Verify the dimension.
    final_shape = dict(datacube.xrdataset.dims)
    assert final_shape[NAME_AZIMUTH] == AZIMUTH_DIM
    assert final_shape[NAME_RANGE] == AZIMUTH_DIM
    assert final_shape[NAME_BAND] == 33

    # verify the content itself - first by checking the nan values
    for i in range(33):
        if i not in [1, 2, 31]:
            if (
                datacube.xarray_datatype == "float32"
                or datacube.xarray_datatype == "float64"
            ):
                assert np.isnan(datacube.xrdataset[NAME_REAL_BAND][i].values[()]).all()
                assert np.isnan(
                    datacube.xrdataset[NAME_COMPLEX_BAND][i].values[()]
                ).all()
            else:
                assert np.all(datacube.xrdataset[NAME_REAL_BAND][i].values[()] == 0)
                assert np.all(datacube.xrdataset[NAME_COMPLEX_BAND][i].values[()] == 0)

            for key in _get_metatdata_keys(datacube):
                assert datacube.get_metadata(key, NAME_REAL_BAND, i) == "None"
                assert datacube.get_metadata(key, NAME_COMPLEX_BAND, i) == "None"

    # verify the content itself
    intersting_id_in_cubes = [1, 2, 31]
    for i, name_file in enumerate(LIST_ORDER_BY_ACQUISITION_DATE):
        hf = h5py.File(os.path.join(SLC_RASTER_DIR, name_file), "r")
        # Extracting raster bands
        s_i_h5 = np.array(hf["s_i"])
        s_q_h5 = np.array(hf["s_q"])
        index_cube = intersting_id_in_cubes[i]
        assert np.array_equal(
            datacube.xrdataset[NAME_REAL_BAND][index_cube].values, s_i_h5
        )
        assert np.array_equal(
            datacube.xrdataset[NAME_COMPLEX_BAND][index_cube].values, s_q_h5
        )

    # check the metadata.
    for key, items in EXPECTED_METADATA_VALUES.items():
        for i, item in enumerate(items):
            index_cube = intersting_id_in_cubes[i]

            assert datacube.get_metadata(key, NAME_REAL_BAND, index_cube) == item
            assert datacube.get_metadata(key, NAME_COMPLEX_BAND, index_cube) == item


def test_temporal_resolution():

    cube_config = CubeConfig()
    cube_config.temporal_resolution = 2
    cube_config.start_date = 20210427
    cube_config.end_date = 20210501
    cube_config.load_config(None)

    datacube = SLCDatacube.build(cube_config, SLC_RASTER_DIR)

    # Verify the dimension.
    final_shape = dict(datacube.xrdataset.dims)
    assert final_shape[NAME_AZIMUTH] == AZIMUTH_DIM
    assert final_shape[NAME_RANGE] == AZIMUTH_DIM
    # 33 days between the begining and the end
    assert final_shape[NAME_BAND] == 3

    # Assert values
    hf = h5py.File(os.path.join(SLC_RASTER_DIR, LIST_ORDER_BY_ACQUISITION_DATE[0]), "r")
    # Extracting raster bands
    s_i_h5 = np.array(hf["s_i"])
    s_q_h5 = np.array(hf["s_q"])
    assert np.array_equal(datacube.xrdataset[NAME_REAL_BAND][0].values, s_i_h5)
    assert np.array_equal(datacube.xrdataset[NAME_COMPLEX_BAND][0].values, s_q_h5)

    if datacube.xarray_datatype == "float32" or datacube.xarray_datatype == "float64":
        assert np.isnan(datacube.xrdataset[NAME_REAL_BAND][1].values[()]).all()
        assert np.isnan(datacube.xrdataset[NAME_COMPLEX_BAND][1].values[()]).all()
        assert np.isnan(datacube.xrdataset[NAME_REAL_BAND][2].values[()]).all()
        assert np.isnan(datacube.xrdataset[NAME_COMPLEX_BAND][2].values[()]).all()
    else:
        assert np.all(datacube.xrdataset[NAME_REAL_BAND][1].values[()] == 0)
        assert np.all(datacube.xrdataset[NAME_COMPLEX_BAND][1].values[()] == 0)
        assert np.all(datacube.xrdataset[NAME_REAL_BAND][2].values[()] == 0)
        assert np.all(datacube.xrdataset[NAME_COMPLEX_BAND][2].values[()] == 0)


def test_slc_datacube_from_list():

    # Load the default config
    cube_config = CubeConfig()
    cube_config.load_config(None)

    list_path = [
        Path(os.path.join(SLC_RASTER_DIR, cur_path))
        for cur_path in [
            "ICEYE_SLC_54549_20210427T215124_hollow_20x20pixels_fake_0.h5",
            "ICEYE_SLC_54549_20210427T215124_hollow_20x20pixels_fake_1.h5",
        ]
    ]
    datacube = SLCDatacube.build_from_list(cube_config, list_path)

    # Verify the dimension.
    final_shape = dict(datacube.xrdataset.dims)
    assert final_shape[NAME_AZIMUTH] == AZIMUTH_DIM
    assert final_shape[NAME_RANGE] == AZIMUTH_DIM
    assert final_shape[NAME_BAND] == len(list_path)

    # verify the content itself
    for i, name_file in enumerate(LIST_ORDER_BY_ACQUISITION_DATE[:2]):
        hf = h5py.File(os.path.join(SLC_RASTER_DIR, name_file), "r")
        # Extracting raster bands
        s_i_h5 = np.array(hf["s_i"])
        s_q_h5 = np.array(hf["s_q"])

        assert np.array_equal(datacube.xrdataset[NAME_REAL_BAND][i].values, s_i_h5)
        assert np.array_equal(datacube.xrdataset[NAME_COMPLEX_BAND][i].values, s_q_h5)

    # check the metadata.
    for key, items in EXPECTED_METADATA_VALUES.items():
        for i, item in enumerate(items[:2]):
            assert datacube.get_metadata(key, NAME_REAL_BAND, i) == item
