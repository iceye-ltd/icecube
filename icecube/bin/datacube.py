#!/usr/bin/env python
"""
datacube class provides a user-friendly interface to access relevant information from the datacubes.
"""
import pickle
import xarray as xr
import numpy as np


def assert_xrdataset(xrdataset):
    if not (isinstance(xrdataset, xr.Dataset)):
        raise ValueError(f"xr.Dataset expected while {type(xrdataset)} was passed")


def assert_xrarray(xrarray):
    if not (isinstance(xrarray, xr.DataArray)):
        raise ValueError(f"xr.DataArray expected while {type(xrarray)} was passed")


class Datacube:
    """
    Datacube core class that provides major functionalities to RW/access icecubes
    Please read documentation here: [http://xarray.pydata.org/en/stable/user-guide/data-structures.html]
    to understand more about data structure of xarray
    """

    def __init__(self, xrdataset=None):
        self.xrdataset = xrdataset

    def read_cube(self, cube_fpath):
        """
        Given a path to datacube in netCDF4 format, read the datacube and returns the instance of class
        :param ciube_fpath: path/to/cube in netCDF4 format.
        returns instance of class with updated dataset attribute
        """
        with xr.open_dataset(cube_fpath) as xds:
            self.xrdataset = xds

        return self

    def get_data_variables(self) -> list:
        """
        Given xr.Dataset, return sequence of data variables
        """
        return list(self.xrdataset.data_vars)

    def set_xrdataset(self, xrdataset: xr.Dataset):
        """
        set xarray dataset as Datacube attribute
        :pararm xrdataest: update dataset attribute of the class, of type xr.Dataset
        """
        self.xrdataset = xrdataset
        return self

    def get_xrarray(self, data_variable: str) -> xr.DataArray:
        """
        Get xr.DataArray given a data-variable name
        :param data_variable: name of a data variable of xr.Dataset
        """
        return self.xrdataset[data_variable]

    def get_dimensions(self) -> dict:
        """
        Get dimension of the associated xr.Dataset
        returns size of xr.Dataset as dict, e.g., {"Azimuth": 100, "Range": 50}
        """
        return dict(self.xrdataset.sizes)

    def get_xrdataset_metadata(self) -> dict:
        """
        Get metadata of xr.Dataset
        :returns attributes of xr.Dataset as dict
        """
        return self.xrdataset.attrs

    def get_xrarray_metadata(self, data_variable) -> dict:
        """
        Get metadata of xr.DataArray
        :param data_variable: name of a data variable of xr.Dataset
        :returns attributes of xr.DataArray as dict
        """
        return self.xrdataset[data_variable].attrs

    @staticmethod
    def get_all_products(xrarray: xr.DataArray) -> list:
        """
        Given a datavariable of xr.Dataset, returns all product-files found
        :param data_variable:
        """
        product_files = xrarray.attrs["product_file"]
        if not (isinstance(product_files, list)):
            product_files = [product_files]

        if not (all(e == "None" for e in product_files)):
            return product_files
        else:
            raise KeyError("product files metadata is missing from the xr.DataArray")

    @staticmethod
    def get_product_index(product_file: str, xrarray: xr.DataArray) -> int:
        """
        Get index of product file from xr.DataArray
        :pararm product_file: name of the product file
        :pararm xrarray: xr.DataArray
        returns index of product file inside xr.DataArray
        """
        if xrarray.attrs == {}:
            raise ValueError(
                f"No metadata found against provided xr.DataArray {xrarray}"
            )

        try:
            return int(xrarray.attrs["product_file"].index(product_file))
        except Exception:
            raise KeyError(
                "product_file: {} is missing from xr.DataArray metadata".format(
                    product_file
                )
            )

    @staticmethod
    def get_metadata_by_product(product_file: str, xrarray: xr.DataArray) -> dict:
        """
        Given xr.DataArray, return metadata from xr.DataArray filtered by product-file
        :pararm product_file: name of the product file
        :pararm xrarray: xr.DataArray
        :returns attributes of product file inside xr.DataArray
        """
        assert_xrarray(xrarray)
        product_index = Datacube.get_product_index(product_file, xrarray)
        product_metadata = {}

        for key, metadata_seq in xrarray.attrs.items():
            product_metadata[key] = metadata_seq[product_index]

        return product_metadata

    @staticmethod
    def get_product_values(product_file: str, xrarray: xr.DataArray):
        """
        Given xr.DataArray, get productfile values. The function looks for raster types, and returns
        np.array otherwise unserializes the dict labels using pickle.loads()
        :pararm product_file: name of the product file
        :pararm xrarray: xr.DataArray
        returns values associated with the product file inside xr.DataArray
        """
        raster_dtypes = ["int8", "uint8", "int16", "uint16", "float32", "float64"]
        assert_xrarray(xrarray)

        # If dtype on of above, raster detected, otherwise pickl.loads() vector labels
        if xrarray.dtype in raster_dtypes:
            return np.array(
                xrarray.values[Datacube.get_product_index(product_file, xrarray)]
            )
        else:
            return pickle.loads(
                xrarray.values[Datacube.get_product_index(product_file, xrarray)]
            )

    @staticmethod
    def get_index_values(index: str, xrarray: xr.DataArray):
        """
        Given xr.DataArray, get_values from the index.
        :pararm product_file: name of the product file
        :pararm xrarray: xr.DataArray
        returns np.array otherwise unserializes the dict labels using pickle.loads()
        """
        raster_dtypes = ["int8", "uint8", "int16", "uint16", "float32", "float64"]
        assert_xrarray(xrarray)

        # If dtype on of above, raster detected, otherwise pickl.loads() vector labels
        if xrarray.dtype in raster_dtypes:
            return np.array(xrarray.values[index])
        else:
            return pickle.loads(xrarray.values[index])

    @staticmethod
    def merge_xrdatasets(
        xrdataset_seq: list, combine_attrs="drop_conflicts"
    ) -> xr.Dataset:
        """
        Merge lists of Datasets together using xt.Dataset.merge
        :param xrdataset_seq: list of xr.Dataset
        :param combine_attrs: String indicating how to combine attrs of the objects being merged
                              {"drop", "identical", "no_conflicts", "drop_conflicts", "override"}, default: "drop_conflicts")

        returns combined xr.Dataset
        """
        xrdataset_merged = None

        for i, xrdataset in enumerate(xrdataset_seq):
            assert_xrdataset(xrdataset)
            if i < 1:
                xrdataset_merged = xrdataset
                continue

            xrdataset_merged = xrdataset_merged.merge(
                xrdataset, combine_attrs=combine_attrs
            )

        return xrdataset_merged

    def to_file(self, output_fpath: str, format="netCDF4"):
        """
        :param output_fpath: path/to/cube.nc where xr.Dataset will be saved
        :param format: File format for the resulting netCDF file,
                      can be {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"}
        """
        if self.xrdataset is None:
            raise Exception("Please reformat the")

        self.xrdataset.to_netcdf(output_fpath, mode="w", format=format)
