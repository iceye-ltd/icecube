#!/usr/bin/env python
"""
Script that reads configuration file for creating datacubues
"""

import json
from datetime import datetime
from icecube.utils.logger import Logger
import os

logger = Logger(os.path.basename(__file__))
import warnings

POSSIBLE_KEYS = [
    "start_date",
    "end_date",
    "min_incidence_angle",
    "max_incidence_angle",
    "temporal_resolution",
    "coregistered",
    "space_overlap",
    "temporal_overlap",
]


class CubeConfig:
    def __init__(self):
        """
        Class reads user configured json file and parses them to make sure generated cube is in accordance
        with user demand.

        Expected Json keys:
            {start_time, end_time, temporal_resolution, incidence_angle, space_overlap, temporal_overlap}
        If keys are not specified, they are set to default values. see CubeConfig._load_defaults_params()

        A sample json looks like this:
        {
            "start_date": 20200101,
            "end_date": 20203101,
            "temporal_resolution": 1,
            "min_incidence_angle": 20,
            "max_incidence_angle": 35,
            "space_overlap": 0,
            "temporal_overlap": 0
        }
        """
        self._load_default_params()  # iniate default user-config

    def _load_default_params(self):
        self.start_date = 19700101
        self.end_date = datetime.strftime(datetime.today(), "%Y%m%d")
        self.min_incidence_angle = 0  # defaults to 0
        self.max_incidence_angle = 90  # defaults to 90
        self.temporal_resolution = 0  # an integer in Days. defaults to 0. If 0, natural timelines of stacks are used.
        self.coregistered = False  # defaults False. Stack is coregistered or not.
        self.space_overlap = False  # (bool) defaults to False. Should only overlapping rasters be considered for cubes (irrelevant if stack is coregistered)
        self.temporal_overlap = True  # (bool) defaults to True. If True, retains rasters from the same dates, drops otherwise

        self.cube_config = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "min_incidence_angle": self.min_incidence_angle,
            "max_incidence_angle": self.max_incidence_angle,
            "temporal_resolution": self.temporal_resolution,
            "coregistered": self.coregistered,
            "space_overlap": self.space_overlap,
            "temporal_overlap": self.temporal_overlap,
        }

    def _validate_user_config(self, user_config: dict):
        """
        Validate if user configurations were properly passed
        """

        if len(user_config) == 0:
            logger.debug("no specific config from the user - take previous values")
            return

        for key in user_config.keys():
            if key not in POSSIBLE_KEYS:
                warnings.warn(
                    f"found un-supported key in user configuration: {key} ",
                    stacklevel=2,
                )
                logger.debug(f"found un-supported key in user configuration: {key}")

        try:
            bool(user_config.get("space_overlap", True))
            bool(user_config.get("temporal_overlap", True))
            bool(user_config.get("coregistered", True))
        except Exception as e:
            logger.error(
                f"space_overlap, temporal_overlap, coregistered should be boolean format - possible value True, 'True', 1 - {str(e)}"
            )
            raise Exception(
                "user config - space_overlap, temporal_overlap, coregistered parameters don't match the expected format"
            )

        try:
            datetime.strptime(str(user_config.get("start_date", "20210402")), "%Y%m%d")
            datetime.strptime(str(user_config.get("end_date", "20210420")), "%Y%m%d")
        except Exception as e:
            logger.error(
                f"start_date, end_date should int or str date %Y%m%d format - possible value 20200201, '20200201' - {str(e)}"
            )
            raise Exception(
                "user config start_date, end_date parameters don't match the expected format"
            )

        try:
            float(user_config.get("min_incidence_angle", "0"))
            float(user_config.get("max_incidence_angle", "0"))
        except Exception as e:
            logger.error(
                f"min_incidence_angle, max_incidence_angle should be float format - possible value 0.3, '0.3' - {str(e)}"
            )
            raise Exception(
                "user config min_incidence_angle, max_incidence_angle parameters don't match the expected "
                "format"
            )

        if not type(user_config.get("temporal_resolution", 1)) == int:
            logger.error(f"temporal_resolution should be int format - possible value 1")
            raise Exception(
                "user config temporal_resolution parameter does not match the expected format"
            )

    def _parse_cube_config(self):
        """
        Once the user configuration has been accepted, parse it to create coherent structure for cube.config i.e.,
        objects, dtypes should remain always same.
        """
        self.cube_config["start_date"] = datetime.strptime(
            str(self.cube_config["start_date"]), "%Y%m%d"
        )
        self.cube_config["end_date"] = datetime.strptime(
            str(self.cube_config["end_date"]), "%Y%m%d"
        )

        self.cube_config["min_incidence_angle"] = float(
            self.cube_config["min_incidence_angle"]
        )
        self.cube_config["max_incidence_angle"] = float(
            self.cube_config["max_incidence_angle"]
        )
        self.cube_config["coregistered"] = bool(self.cube_config["coregistered"])

        self.cube_config["space_overlap"] = bool(self.cube_config["space_overlap"])
        self.cube_config["temporal_overlap"] = bool(
            self.cube_config["temporal_overlap"]
        )
        self.cube_config["temporal_resolution"] = int(
            self.cube_config["temporal_resolution"]
        )

    def _assert_cube_config(self):
        """
        Create checks to avoid non-logical cube configuration.
        """
        assert (
            self.cube_config["end_date"] >= self.cube_config["start_date"]
        ), "end date must be >= equal to start date"
        assert (
            self.cube_config["max_incidence_angle"]
            >= self.cube_config["min_incidence_angle"]
        ), "max_incidence_angle must be >= min_incidence_angle"

    def get_config_dict(self):

        self.cube_config = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "min_incidence_angle": self.min_incidence_angle,
            "max_incidence_angle": self.max_incidence_angle,
            "temporal_resolution": self.temporal_resolution,
            "coregistered": self.coregistered,
            "space_overlap": self.space_overlap,
            "temporal_overlap": self.temporal_overlap,
        }
        self._parse_cube_config()
        self._assert_cube_config()
        return self.cube_config

    def load_config(self, json_fpath: None):
        user_config = {}
        try:
            if json_fpath is not None:
                with open(json_fpath) as f:
                    user_config = json.load(f)

        except Exception as e:
            logger.error(f"impossible to load the json - {json_fpath}")
            raise e

        self._validate_user_config(user_config)

        # replace user specified key values with default values
        for k, v in self.cube_config.items():
            if k in user_config:
                # update attr and config
                setattr(self, k, user_config[k])

        return self.get_config_dict()


def sample_workflow():
    from icecube.project_variables import cube_config_fpath

    cube_config = CubeConfig()
    cube_config.load_config(cube_config_fpath)
