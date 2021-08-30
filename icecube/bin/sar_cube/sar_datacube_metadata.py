#!/usr/bin/env python
"""
Metadata class handler for SAR images
"""
import pandas as pd
from typing import Tuple, List
import shapely
import os
import numpy as np
from pathlib import Path

# Local import
from icecube.utils.logger import Logger
from icecube.utils.metadata_crawler import metadata_crawler, metadata_crawler_list

logger = Logger(os.path.basename(__file__))

from icecube.bin.datacube_variables import DEFAULT_FILL_VALUE


class SARDatacubeMetadata:
    """
    Represents the metadata of the datacube - the metadata_df is the one used to create the datacube
    Please follow the documentation and the expected shape to create your datacube
    """

    SUPPORTED_VARIABLES = [
        "product_file",
        "product_fpath",
        "incidence_center",
        "look_side",
        "orbit_direction",
        "extent",
        "acquisition_date",
        "acquisition_time",
        "number_of_azimuth_samples",
        "number_of_range_samples",
    ]

    def __init__(
        self,
        cube_config,
        metadata_df: pd.DataFrame = None,
        master_id: str = None,
        fill_value: int = DEFAULT_FILL_VALUE,
        xarray_datatype="default",
    ):

        self.cube_config = cube_config
        self._unpack_cube_config()
        self.metadata_df = metadata_df
        self.master_id = master_id
        self.fill_value = fill_value
        self.xarray_datatype = xarray_datatype

    def _unpack_cube_config(self):
        self.start_date = self.cube_config["start_date"]
        self.end_date = self.cube_config["end_date"]
        self.min_incidence_angle = self.cube_config["min_incidence_angle"]
        self.max_incidence_angle = self.cube_config["max_incidence_angle"]
        self.temporal_resolution = self.cube_config["temporal_resolution"]
        self.coregistered = self.cube_config["coregistered"]
        self.space_overlap = self.cube_config["space_overlap"]
        self.temporal_overlap = self.cube_config["temporal_overlap"]
        self.fill_value = None
        self.fill_method = None

    def _crawl_metadata(self, raster_dir, product_type):
        return metadata_crawler(
            raster_dir,
            product_type,
            SARDatacubeMetadata.SUPPORTED_VARIABLES,
            recursive=False,
        )

    def _crawl_metadata_list(self, list_images: List[Path]):
        return metadata_crawler_list(
            [str(cur_path) for cur_path in list_images],
            SARDatacubeMetadata.SUPPORTED_VARIABLES,
        )

    def compute_metdatadf_from_list(self, list_path: List[Path]):
        """
        The function will go throuh the metadata.
        """

        logger.info(f"Building the metadata from the list {list_path}")

        # check the input
        if list_path is None or len(list_path) == 0:
            raise Exception(f"Please provide a correct input - {list_path}")

        ext = list_path[0].suffix
        for cur_path in list_path:
            if not cur_path.exists() and not cur_path.is_file():
                raise Exception(f"File {str(cur_path)} isn't a correct path")
            if cur_path.suffix != ext:
                raise Exception(
                    f"All files should have the same extensions"
                    f"{str(cur_path)} is not  {ext}"
                )

        self.metadata_df = self._crawl_metadata_list(list_path)

        logger.debug(f"length metadata from the directory {len(self.metadata_df)}")

        # The order here is important as it'll used to build the datacube.
        self.metadata_df = self.sort_df_by_date()

        # Prune dataframe according to requested days.
        self.metadata_df = self.select_requested_dates()
        logger.debug(f"length metadata after filter by date {len(self.metadata_df)}")

        # Prune metadata df w.r.t incidence anngles
        self.metadata_df = self.select_requested_angles()
        logger.debug(
            f"length metadata after filter requested angle {len(self.metadata_df)}"
        )

        # If two rasters are at the same date, the one with greater timestamp will be kept
        if not (
            self.temporal_overlap or self.metadata_df["acquisition_date"].is_unique
        ):
            self.metadata_df = self.prune_temporal_overlap()

        if self.space_overlap and not (self.coregistered):
            self.metadata_df = self.select_overlapping_rasters()

        # Add empty row as reference for the temporal resolution.
        if bool(self.temporal_resolution):
            self.metadata_df = self.set_temporal_resolution(
                method=self.fill_method, fill_value=self.fill_value
            )
        self.assert_non_empty_dataframe()
        return self

    def compute_metdatadf_from_folder(self, raster_dir: str, product_type: str):
        """
        The function will go throuh the metadata.
        """
        logger.info(
            f"Building the metadata from the folder {raster_dir} using {product_type}"
        )
        self.metadata_df = self._crawl_metadata(raster_dir, product_type)
        logger.debug(f"length metadata from the directory {len(self.metadata_df)}")

        # The order here is important as it'll used to build the datacube.
        self.metadata_df = self.sort_df_by_date()

        # Prune dataframe according to requested days.
        self.metadata_df = self.select_requested_dates()
        logger.debug(f"length metadata after filter by date {len(self.metadata_df)}")

        # Prune metadata df w.r.t incidence anngles
        self.metadata_df = self.select_requested_angles()
        logger.debug(
            f"length metadata after filter requested angle {len(self.metadata_df)}"
        )

        # If two rasters are at the same date, the one with greater timestamp will be kept
        if not (
            self.temporal_overlap or self.metadata_df["acquisition_date"].is_unique
        ):
            self.metadata_df = self.prune_temporal_overlap()

        if self.space_overlap and not (self.coregistered):
            self.metadata_df = self.select_overlapping_rasters()

        # Add empty row as reference for the temporal resolution.
        if bool(self.temporal_resolution):
            self.metadata_df = self.set_temporal_resolution(
                method=self.fill_method, fill_value=self.fill_value
            )

        self.assert_non_empty_dataframe()
        return self

    def assert_non_empty_dataframe(self):
        if self.metadata_df.empty:
            raise ValueError(
                "No rasters found against given configuration. Please check user-configuration."
            )

    def sort_df_by_date(self):
        self.metadata_df["Date"] = pd.to_datetime(self.metadata_df["acquisition_date"])
        metadata_df = self.metadata_df.sort_values(by=["Date"])
        return metadata_df.drop(["Date"], axis=1).reset_index(drop=True)

    def select_requested_dates(self):
        acq_date_df = pd.to_datetime(
            self.metadata_df["acquisition_date"], format="%Y%m%d"
        )
        # slice from head w.r.t start date and slice it again from tail w.r.t end date
        df_start = acq_date_df[acq_date_df >= self.start_date]
        df_start2end = df_start[df_start <= self.end_date]

        return self.metadata_df.iloc[df_start2end.index].reset_index(drop=True)

    def select_requested_angles(self):
        indexed_df = self.metadata_df[
            (self.metadata_df["incidence_center"] >= self.min_incidence_angle)
            & (self.metadata_df["incidence_center"] <= self.max_incidence_angle)
        ]
        return self.metadata_df.iloc[indexed_df.index].reset_index(drop=True)

    def prune_temporal_overlap(self):

        # If user has specified don't-care-condition for bool or dates are unique, return same df.
        # otherwise group similar dates toegether, and delete rows that don't have the maximum timestamp for same dates.
        rows_to_drop = []

        for _, same_date_acquisition_df in self.metadata_df.groupby("acquisition_date"):
            if len(same_date_acquisition_df) == 1:
                continue

            max_index = pd.to_numeric(
                same_date_acquisition_df["acquisition_time"]
            ).idxmax()
            drop_indices = same_date_acquisition_df.index.values.tolist()
            drop_indices.remove(int(max_index))
            rows_to_drop += drop_indices

        return self.metadata_df.drop(rows_to_drop, axis=0)

    def select_overlapping_rasters(self):
        """
        Select only rasters that have an overlap with each other.
        By defaults primary polygnon has overap with itself.
        """
        # If latitude, longitude points are provided by the user, then
        # we need to think how that integretes into this too.
        min_index = pd.to_numeric(self.metadata_df["acquisition_date"]).idxmin()
        primary_polygon = shapely.wkt.loads(self.metadata_df.iloc[min_index]["extent"])
        secondary_polygons = self.metadata_df["extent"].apply(shapely.wkt.loads)
        intersect_indices = secondary_polygons.apply(primary_polygon.intersects)
        overlapping_rasters = self.metadata_df.iloc[intersect_indices.values.tolist()]

        if (overlapping_rasters.shape[0]) == 1:
            logger.warning(
                "No secondary raster(s) found with current configuration that has space overlap with primary raster",
            )

        return overlapping_rasters

    def set_temporal_resolution(self, method=None, fill_value=None):

        acq_date_df = pd.to_datetime(
            self.metadata_df["acquisition_date"], format="%Y%m%d"
        )
        metadata_df_date_indexed = self.metadata_df.set_index(acq_date_df)

        # If duplicates found, use must specify on
        if metadata_df_date_indexed.index.duplicated().any():
            raise ValueError(
                f"Duplicate entries found for configured temporal resolution: {self.temporal_resolution}, please set temporal_overlap to false"
            )

        sampled_metadata_df = metadata_df_date_indexed.asfreq(
            freq=str(self.temporal_resolution) + "D",
            method=method,
            fill_value=np.nan,
        )

        # set the index to df to user defined start/end dates.
        datetime_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=str(self.temporal_resolution) + "D",
        )

        sampled_metadata_df = sampled_metadata_df.reindex(datetime_range)
        sampled_metadata_df = sampled_metadata_df.reset_index()
        sampled_metadata_df = sampled_metadata_df.drop(["acquisition_date"], axis=1)
        sampled_metadata_df = sampled_metadata_df.rename(
            columns={"index": "acquisition_date"}
        )
        return sampled_metadata_df

    def get_master_shape(self) -> Tuple[int, int]:
        """

        Returns:
            Shape of the master frame, if self.master_id is None, first frame with a shape
        """
        index_master = self.metadata_df.number_of_azimuth_samples.first_valid_index()
        intersting_row = self.metadata_df.iloc[index_master]
        return int(intersting_row["number_of_azimuth_samples"]), int(
            intersting_row["number_of_range_samples"]
        )

    def get_lenght(self) -> int:
        """
            Return the number of layers.
        Returns:
            The number of row - each row being a layer in the datacube
        """
        return len(self.metadata_df)

    def visualize(self):
        """
        pre-visualize the datacube
        """
        pass

    def get_dummy_data(self):
        return {
            "PRODUCT_FILE": "NA",
            "DESCRIPTION": "temporal_gap",
            "fill_value": self.fill_value,
        }
