import datetime
import os
from pathlib import Path
import pytest

# Local import
from icecube.bin.config import CubeConfig

JSON_CONFIG_DIR = os.path.join(Path(__file__).parent, "resources", "json_config")


def test_load_config():
    cube_config = CubeConfig()
    cube_config_dict = cube_config.load_config(
        os.path.join(JSON_CONFIG_DIR, "config_use_case1.json")
    )
    print(cube_config_dict)

    assert cube_config.temporal_resolution == 1
    assert float(cube_config.min_incidence_angle) == 20
    assert float(cube_config.max_incidence_angle) == 21
    assert bool(cube_config.coregistered)
    assert bool(cube_config.space_overlap)
    assert bool(cube_config.temporal_overlap)
    assert str(cube_config.start_date) == "20200402"
    assert str(cube_config.end_date) == "20210420"

    assert cube_config_dict["temporal_resolution"] == 1
    assert cube_config_dict["min_incidence_angle"] == 20
    assert cube_config_dict["max_incidence_angle"] == 21
    assert cube_config_dict["coregistered"]
    assert cube_config_dict["space_overlap"]
    assert cube_config_dict["temporal_overlap"]
    assert cube_config_dict["start_date"] == datetime.datetime(2020, 4, 2)
    assert cube_config_dict["end_date"] == datetime.datetime(2021, 4, 20)


def test_load_config():
    cube_config = CubeConfig()
    cube_config_dict = cube_config.load_config(
        os.path.join(JSON_CONFIG_DIR, "config_use_case2.json")
    )
    assert cube_config_dict["start_date"] == datetime.datetime(2020, 4, 2)
    assert cube_config_dict["end_date"] == datetime.datetime(2021, 4, 20)


def test_wrong_type_parameter():
    with pytest.raises(Exception) as e_info:
        cube_config = CubeConfig()
        cube_config.load_config(os.path.join(JSON_CONFIG_DIR, "config_use_case3.json"))

    assert "temporal_resolution" in str(e_info)
