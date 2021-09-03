#!/usr/bin/env python

"""
Common utilities that can be widely used across the package
"""

import numpy as np
import os
import glob
from itertools import groupby
import sys
import time

from icecube.utils.logger import Logger

logger = Logger(os.path.basename(__file__))


class DirUtils:
    @staticmethod
    def get_dir_files(local_dir, fext=None):
        """
        scrape the given directory for all files or files with certain extension.
        fext: str or list of strings.
        """    
        local_dir = os.path.join(local_dir, "")
        fpaths = []  # contains names of all images.

        if fext and not(isinstance(fext, list)):
            fext = list(fext)

        if fext:
            for ext in fext:
                ext = ext.replace("*", "")  # if asterik was passed by mistake.
                fpaths.extend(glob.glob(local_dir + "*" + ext))
        else:
            fpaths.extend(glob.glob(local_dir + "*"))

        fnames = [p.replace(local_dir, "") for p in fpaths]

        return fnames, fpaths


def get_dummy_metadata():
    """
    Incase of a temporal gap, the following metadata is used
    """
    return dict({"product_file": "None"})


def get_product_metadata(product_file):
    """
    Metadata appended for labels xr.DataArray
    """
    return dict({"product_file": product_file})


def assert_metadata_exists(metadata_df):
    """
    Make sure that metadata is non-empty
    Returns True if non-empty otherwise raises exception
    """
    for df_col in metadata_df.columns:
        if metadata_df[df_col].isnull().values.all():
            raise ValueError(
                "No valid values found for pd.Series: {} for pd.column:{} against given configuration".format(
                    metadata_df[df_col], df_col
                )
            )

    return True


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def match_dict_key(d, value):
    """
    get key by matching value of a dictionary.
    d: dictionary of type dict.
    value: value to match
    returns: key of matched value.
    """
    for k, v in d.items():
        if v == value:
            return k

    print(f"value: {value} not found in dictionary")


def draw_progress_bar(percent, barLen=50):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write(f"[ {progress} ] {percent * 100:.2f}%")
    sys.stdout.flush()


def measure_time(func):
    """
    :param func:
    :return:
    """

    def time_it(*args, **kwargs):
        time_started = time.time()
        return_value = func(*args, **kwargs)
        time_elapsed = time.time()
        logger.info(
            "{execute} running time is {sec} seconds".format(
                execute=func.__name__,
                sec=round(time_elapsed - time_started, 4),
            )
        )
        return return_value

    return time_it


class NumpyEncoder:
    """Special json encoder for numpy types"""

    @classmethod
    def encode(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return str(obj.tolist())
        else:
            return obj


def get_slc_metadata(h5f):
    """
    Read metadata from SLC header

    Args:
        opened_h5_file: SLC h5 file opened as a h5py File
    """

    meta_dict = {}

    # h5py doesn't read .keys() correctly, but returns a generator. So parsing to list to get all keys.
    key_list = [x for x in h5f.keys()]

    # Check if there already is a list of bands
    if "bands" in key_list:
        non_meta_keys = h5f["bands"]
    else:
        non_meta_keys = ["s_i", "s_q"]

    # Splines are not passed as they are not necessary. We just skip them for now.
    # This doesn't not effect the SLC in any meaningfull way
    additional_bands_to_skip_for_now = [
        "RPC",
        "height_spline",
        "lat_spline",
        "lon_spline",
    ]
    for key in key_list:
        if key not in non_meta_keys + additional_bands_to_skip_for_now:

            h5_meta_val = h5f[key]

            if type(h5_meta_val[()]) == bytes:
                # h5py 3.0+ introduces str stored as bytes. We use .asstr() to read it (but breaks with h5py < 2.10)
                meta_dict[key] = np.array_str(np.array(h5_meta_val.asstr()[()]))

            elif type(h5_meta_val[()]) == np.bytes_:
                # This is most likely a date
                meta_dict[key] = np.array_str(
                    np.squeeze(np.char.decode(h5_meta_val, "utf-8"))
                )

            else:
                # We're handling a numpy array so we should be good to go
                meta_dict[key] = np.array_str(np.array(h5_meta_val[()]))

                # RPCs are nested under "RPC/" in the h5 thus need to be parsed in a specific manner
    RPC_source = h5f["RPC"]

    for key, val in RPC_source.items():
        meta_dict[f"RPC_{key}"] = np.array(val, dtype=np.float32)

    return meta_dict
