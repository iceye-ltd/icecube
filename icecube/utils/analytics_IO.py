"""
Input and output functions for the ICEYE rasters
"""

import os.path
import warnings

import h5py
import xmltodict
import numpy as np
import rasterio


def load_tiff(path):
    """
    Load a generic tiff with rasterio.
    """

    assert isinstance(path, str), (
        'Expected the input "path" to be a string, instead a %s datatype variable was input.'
        % type(path)
    )
    assert path.endswith(".tif") or path.endswith(
        ".tiff"
    ), "Expected the input path to have the suffix .tif or .tiff."

    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    with rasterio.open(path) as rasterio_reader:
        I = np.squeeze(rasterio_reader.read())

    warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

    return I


def load_ICEYE_GRD(path, load_metadata=False):

    assert isinstance(path, str), (
        "Did not understand input path, a str was expected but a %s datatype variable was input."
        % type(path)
    )
    assert path.endswith(".tif") or path.endswith(
        ".tiff"
    ), "A .tif or .tiff was expected but something else was input."

    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    container = rasterio.open(path)

    warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

    if not container:  # Could not be read.
        _raise_rasterio_IOError(path)

    if load_metadata:
        return container, load_ICEYE_metadata(path)
    else:
        return container


def load_ICEYE_SLC(path, load_metadata=False):
    """
    Load the ICEYE h5 as a rasterio datasetreader. Read only.

    Note that the metadata list does not load properly with gdal. Use h5py as
    used in load_metadata to gain access to all the metadata from either the
    .h5 (full list) or .xml (subset, still got the image locations etc).
    You can enable returning metadata with the boolean flag load_metadata.
    """

    return load_ICEYE_h5(path, load_metadata)


def load_ICEYE_h5(path, load_metadata=False):
    """
    Load the ICEYE h5 as a rasterio datasetreader. Read only.

    Note that the metadata list does not load properly with gdal. Use h5py as
    used in load_metadata to gain access to all the metadata from either the
    .h5 (full list) or .xml (subset, still got the image locations etc).
    You can enable returning metadata with the boolean flag load_metadata.
    """

    assert isinstance(path, str), (
        "Did not understand input path, a str was expected but a %s datatype variable was input."
        % type(path)
    )
    assert path.endswith(".h5"), "a .h5 was expected but something else was input."

    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    container = rasterio.open(path)

    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    if not container:  # Could not be read.
        _raise_rasterio_IOError(path)

    if load_metadata:
        return container, load_ICEYE_metadata(path)
    else:
        return container


def extract_complex_channels(path_or_rasterio_h5):
    """
    Draw the s_i and s_q components from an ICEYE h5. Returns two rasterio datasets.
    """

    if isinstance(path_or_rasterio_h5, str):  # assume it's a path, try to load.
        rasterio_h5 = load_ICEYE_SLC(path_or_rasterio_h5)
    elif not isinstance(path_or_rasterio_h5, rasterio.io.DatasetReader):
        raise ValueError(
            'Did not understand input "path_or_rasterio_h5", either a str or a rasterio.io.DatasetReader was expected but a %s datatype variable was input.'
            % type(path_or_rasterio_h5)
        )
    else:
        rasterio_h5 = path_or_rasterio_h5

    subdataset_names = rasterio_h5.subdatasets

    # Find the subdatasets with the SLC data.
    i = 0
    s_i_index = None
    s_q_index = None
    for name in subdataset_names:  # name is tuple (location, description)
        if name.endswith("//s_i"):
            s_i_index = i
        elif name.endswith("//s_q"):
            s_q_index = i
        i += 1

    # Check that both were found.
    if s_i_index is None:
        raise FileNotFoundError(
            'Could not find subdataset "s_i" from input file, aborting.'
        )
    elif s_q_index is None:
        raise FileNotFoundError(
            'Could not find subdataset "s_q" from input file, aborting.'
        )

    # Load them as new gdal files
    s_i_name = subdataset_names[s_i_index]
    s_q_name = subdataset_names[s_q_index]

    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    s_i_data = rasterio.open(s_i_name)
    s_q_data = rasterio.open(s_q_name)

    warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

    if not s_i_data:
        raise OSError('Failed to load "%s", aborting.' % s_i_name)
    if not s_q_data:
        raise OSError('Failed to load "%s", aborting.' % s_q_name)

    return s_i_data, s_q_data


def load_SLC_intensity(datasetreader):

    return read_SLC_intensity(datasetreader)


def read_SLC_intensity(datasetreader):

    s_i, s_q = extract_complex_channels(datasetreader)
    intensity = complex2intensity(s_i, s_q)

    return intensity


def load_SLC_amplitude(datasetreader):

    return read_SLC_amplitude(datasetreader)


def read_SLC_amplitude(datasetreader):

    s_i, s_q = extract_complex_channels(datasetreader)
    amplitude = complex2amplitude(s_i, s_q)

    return amplitude


def complex2amplitude(s_i, s_q):
    """
    Transforms the s_i and s_q images into a single amplitude image.
    Assumes hdf5 Dataset or numpy array inputs.
    """

    s_i, s_q = _parse_complex_channels(s_i, s_q)

    # return np.sqrt(s_i**2, s_q**2). Doing it piecewise in the hope it saves some memory,
    # had this function run out of memory on my 32GB machine a few times.

    s_i **= 2
    s_q **= 2

    s_i += s_q
    s_q = None

    return np.sqrt(s_i)


def complex2intensity(s_i, s_q):

    s_i, s_q = _parse_complex_channels(s_i, s_q)

    s_i **= 2
    s_q **= 2

    return s_i + s_q


def _parse_complex_channels(s_i, s_q):

    assert type(s_i) == type(s_q)

    if isinstance(s_i, rasterio.io.DatasetReader):

        s_i = s_i.read(out_dtype=rasterio.float32)
        s_q = s_q.read(out_dtype=rasterio.float32)

        s_i = np.squeeze(s_i)  # The arrays are read as (1,x,y)
        s_q = np.squeeze(s_q)

    elif isinstance(s_i, np.ndarray):
        if s_i.dtype != np.float32:
            s_i = np.float32(s_i)
            s_q = np.float32(s_q)
    else:  # assume hdf5
        s_i = s_i[:, :].astype(np.float32)  # hdf5 Dataset -> numpy array
        s_q = s_q[:, :].astype(np.float32)  # int16 -> float32 for the power and sqrt.

    return s_i, s_q


def load_kml(kml_path):

    with open(kml_path) as fd:
        kml_dict = xmltodict.parse(fd.read())

    polygon_str = kml_dict["kml"]["Document"]["Placemark"]["Polygon"][
        "outerBoundaryIs"
    ]["LinearRing"]["coordinates"]

    polygon_list = polygon_str.split(" ")

    n = len(polygon_list)
    lat = np.empty(n)
    lon = np.empty(n)
    height = np.empty(n)
    for i in range(0, n):
        lon_str, lat_str, height_str = polygon_list[i].split(",")
        lon[i], lat[i], height[i] = (
            np.float64(lon_str),
            np.float64(lat_str),
            np.float64(height_str),
        )

    return lat, lon, height


def load_DEM(DEM_path):
    """
    Loads a DEM as a rasterio dataset. It's expected to be a geotiff with geocoding.
    """

    DEM = rasterio.open(DEM_path)
    if not DEM:
        _raise_rasterio_IOError(DEM_path)
    else:
        return DEM


def parse_slc_rpc_to_meta_dict(RPC_source: h5py.File, meta_dict):
    """
    Parsing SLC metadata to dict.
    The keys are generated in a manner allowing to create subgroups when saved in `.h5`
    Args:
        RPC_source: `h5py` File containing RPCs
        meta_dict: metadata dictionary to update with RPC metadata
    Return:
        rpc_dict: RPC Dictionary containing all RPC fields as numpy arrays
    """
    rpc_dict = {}

    for key, val in RPC_source.items():
        rpc_dict[key] = np.array(val, dtype=np.float32)

    return rpc_dict


def read_SLC_metadata(h5_io):
    meta_dict = {}

    # h5py doesn't read .keys() correctly, but returns a generator. So parsing to list to get all keys.
    key_list = [x for x in h5_io.keys()]

    # Check if there already is a list of bands
    if "bands" in key_list:
        non_meta_keys = h5_io["bands"]
    else:
        non_meta_keys = ["s_i", "s_q"]
        meta_dict["bands"] = non_meta_keys

    additional_bands_to_skip_for_now = [
        "RPC",
        "height_spline",
        "lat_spline",
        "lon_spline",
    ]
    for key in key_list:
        if key not in non_meta_keys + additional_bands_to_skip_for_now:

            h5_meta_val = h5_io[key][()]

            if type(h5_meta_val) == bytes or type(h5_meta_val) == np.bytes_:
                # Strings need to be decoded to utf-8 in order to be read correctly in h5py
                meta_dict[key] = h5_meta_val.decode("utf-8")

            else:
                # We're handling a numpy array so we should be good to go
                meta_dict[key] = np.array(h5_meta_val[()])

    # RPCs are nested under "RPC/" in the h5 thus need to be parsed in a specific manner
    RPC_source = h5_io["RPC"]
    meta_dict["RPC"] = parse_slc_rpc_to_meta_dict(
        RPC_source=RPC_source, meta_dict=meta_dict
    )

    return meta_dict


def correct_grd_metadata_key(original_key: str) -> str:
    """
    Change an upper case GRD key to it's SLC metadata equivalent.
    By default this is uppercasing all keys; otherwise if the value in the
    `special_keys` dict will be used.

    Args:
        original_key: input metadata key
        special_keys: dictionary of special keys that require more than just lower case

    Return:
        corrected_key: corrected key name
    """

    special_keys = {
        "POSX": "posX",
        "POSY": "posY",
        "POSZ": "posZ",
        "VELX": "velX",
        "VELY": "velY",
        "VELZ": "velZ",
    }
    if original_key in special_keys:
        corrected_key = special_keys[original_key]

    else:
        corrected_key = original_key.lower()

    return corrected_key


def read_GRD_metadata(grd_fpath):
    with rasterio.open(grd_fpath) as file:

        metadata = file.tags()
        metadata = {
            correct_grd_metadata_key(key): value for key, value in metadata.items()
        }

        # Also the numeric variables are output as strings. Fix it.
        metadata = _fix_GRD_metadata_datatypes(metadata, _expected_datatypes("GRD"))

        # Load a list of the keys that should be found and a list of which were found.
        expected_keys = _expected_keys("GRD")
        found_keys = metadata.keys()

        # If something is missing mark it with None:
        for var_name in found_keys:
            if var_name not in expected_keys:
                metadata.update({var_name: None})

        metadata = _parse_GRD_RPC(metadata, file)

    # Turn the whining back on.
    warnings.filterwarnings(
        "default",
        "Dataset has no geotransform set",
        category=rasterio.errors.NotGeoreferencedWarning,
    )

    return metadata


def load_ICEYE_metadata(path):
    """
    Load metadata

    h5 has both the SLC images and metadata. gdal/rasterio seems to break a
    lot of the metadata fields, so using h5py or xmltodict for .xml.
    """

    # rasterio is pretty whiny if the image is not properly geocoded (such as any of our GRDs).
    # Turn the warnings off for a while.

    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

    # sanity check
    if not isinstance(path, str):
        raise TypeError(
            'Did not understand input "path", a str was expected but a %s datatype variable was input.'
            % type(path)
        )
    elif not os.path.isfile(path):
        raise FileNotFoundError("No file named %s was found, aborting" % path)

    if path.endswith(".h5"):  # Assume SLC
        """
        h5 has both the SLC images and metadata. gdal/rasterio seems to break a
        lot of the metadata fields on the SLC case, so using h5py.

        Note:: that the SLC reader is not closed unlike the GRD. This is by design, I'd rather
        pass pointers than vectors with tens of thousands of elements. Only the datetimes
        are converted from bytedata and read into the dict for compatability reasons.
        """
        return read_SLC_metadata(h5py.File(path, "r"))

    elif path.endswith(".tif") or path.endswith(".tiff"):
        return read_GRD_metadata(path)

    elif not isinstance(path, str):
        raise TypeError(
            'Could not understand input "path", a str was expected but a %s datatype variable was input.'
            % type(path)
        )
    else:
        raise ValueError(
            'Could not understand input "path", either a .h5, .tif, .tiff or a .xml was expected but %s was input.'
            % path
        )


def _expected_keys(product_type):
    """
    Aux function. Contains the most current lists of keys we expect there to be in the different forms of metadata.
    """
    if product_type == "SLC":

        expected_keys = (
            "acquisition_end_utc",
            "acquisition_mode",
            "acquisition_prf",
            "acquisition_start_utc",
            "angX",
            "angY",
            "angZ",
            "ant_elev_corr_flag",
            "antenna_pattern_compensation",
            "avg_scene_height",
            "azimuth_ground_spacing",
            "azimuth_looks",
            "azimuth_time_interval",
            "calibration_factor",
            "carrier_frequency",
            "chirp_bandwidth",
            "chirp_duration",
            "coord_center",
            "coord_first_far",
            "coord_first_near",
            "coord_last_far",
            "coord_last_near",
            "dc_estimate_coeffs",
            "dc_estimate_poly_order",
            "dc_estimate_time_utc",
            "doppler_rate_coeffs",
            "doppler_rate_poly_order",
            "first_pixel_time",
            "fsl_compensation",
            "geo_ref_system",
            "local_incidence_angle",
            "look_side",
            "mean_earth_radius",
            "mean_orbit_altitude",
            "number_of_azimuth_samples",
            "number_of_dc_estimations",
            "number_of_range_samples",
            "number_of_state_vectors",
            "orbit_absolute_number",
            "orbit_direction",
            "orbit_relative_number",
            "orbit_repeat_cycle",
            "polarization",
            "posX",
            "posY",
            "posZ",
            "processing_prf",
            "processing_time",
            "processor_version",
            "product_level",
            "product_name",
            "product_type",
            "range_looks",
            "range_sampling_rate",
            "range_spread_comp_flag",
            "sample_precision",
            "satellite_name",
            "slant_range_spacing",
            "state_vector_time_utc",
            "total_processed_bandwidth_azimuth",
            "velX",
            "velY",
            "velZ",
            "window_function_azimuth",
            "window_function_range",
            "zerodoppler_end_utc",
            "zerodoppler_start_utc",
        )

    elif product_type == "GRD":

        expected_keys = tuple(_expected_datatypes("GRD"))

    elif product_type == "xml":
        raise NotImplementedError(
            "Ambiguous functionality, the .xml parsing structure does not work as expected. Fix the code before proceeding."
        )
        # A subset of it works, but some variables are missing and some are hidden under a hierarchical structure. You'll need to finish writing the
        # wrapper if you want to use this.

        expected_keys = (
            "Orbit_State_Vectors",
            "Doppler_Centroid_Coefficients",
            "Doppler_Rate",
            "product_name",
            "product_type",
            "product_level",
            "satellite_name",
            "acquisition_mode",
            "look_side",
            "processing_time",
            "processor_version",
            "acquisition_start_utc",
            "acquisition_end_utc",
            "zerodoppler_start_utc",
            "zerodoppler_end_utc",
            "first_pixel_time",
            "number_of_azimuth_samples",
            "number_of_range_samples",
            "orbit_repeat_cycle",
            "orbit_relative_number",
            "orbit_absolute_number",
            "orbit_direction",
            "sample_precision",
            "polarization",
            "azimuth_looks",
            "range_looks",
            "slant_range_spacing",
            "azimuth_ground_spacing",
            "acquisition_prf",
            "processing_prf",
            "carrier_frequency",
            "azimuth_time_interval",
            "range_sampling_rate",
            "chirp_bandwidth",
            "chirp_duration",
            "total_processed_bandwidth_azimuth",
            "window_function_range",
            "window_function_azimuth",
            "range_spread_comp_flag",
            "ant_elev_corr_flag",
            "number_of_dc_estimations",
            "dc_estimate_poly_order",
            "doppler_rate_poly_order",
            "geo_ref_system",
            "avg_scene_height",
            "mean_orbit_altitude",
            "mean_earth_radius",
            "coord_first_near",
            "coord_first_far",
            "coord_last_near",
            "coord_last_far",
            "coord_center",
            "incidence_near",
            "incidence_far",
            "calibration_factor",
        )

    elif not isinstance(product_type, str):
        raise TypeError(
            'Did not understand input "product_type", a str was expected but a %s datatype variable was input.'
            % type(product_type)
        )
    else:
        raise ValueError(
            'Did not understand input "product_type", either "SLC", "GRD" or "xml" was expected but %s was input.'
            % product_type
        )

    return expected_keys


def _expected_datatypes(product_type):
    """
    Aux function. Contains the most current lists of keys we expect there to be in the different forms of metadata.
    """
    if product_type == "SLC":
        # Only the datetimes need to be parsed.

        expected_dtypes = {
            "acquisition_start_utc": "parse_datetime_single",
            "acquisition_end_utc": "parse_datetime_single",
            "dc_estimate_time_utc": "parse_datetime_single",
            "first_pixel_time_utc": "parse_datetime_single",
            "state_vector_time_utc": "parse_datetime_vect",
            "zerodoppler_start_utc": "parse_datetime_single",
            "zerodoppler_end_utc": "parse_datetime_single",
        }

    elif product_type == "GRD":
        # All the fields need to be parsed, so all the datatypes are input.

        expected_dtypes = {
            "acquisition_end_utc": "parse_datetime_single",  # single datetime
            "acquisition_mode": str,
            "acquisition_prf": float,
            "acquisition_start_utc": str,
            "ant_elev_corr_flag": bool,
            "area_or_point": str,
            "avg_scene_height": float,
            "azimuth_spacing": float,
            "azimuth_look_bandwidth": float,
            "azimuth_look_overlap": float,
            "azimuth_looks": int,
            "azimuth_time_interval": float,
            "calibration_factor": float,
            "carrier_frequency": float,
            "chirp_bandwidth": float,
            "chirp_duration": float,
            "coord_center": "parse_float_vect",  # 1d vect of floats, needs to be parsed
            "coord_first_far": "parse_float_vect",
            "coord_first_near": "parse_float_vect",
            "coord_last_far": "parse_float_vect",
            "coord_last_near": "parse_float_vect",
            "dc_estimate_coeffs": "parse_float_vect",
            "dc_estimate_poly_order": int,
            "dc_estimate_time_utc": "parse_datetime_vect",  # datetime vector
            "dc_reference_pixel_time": float,
            "doppler_rate_coeffs": "parse_float_vect",
            "doppler_rate_poly_order": int,
            "doppler_rate_reference_pixel_time": float,
            "gcp_terrain_model": str,
            "geo_ref_system": str,
            "grsr_coefficients": "parse_float_vect",
            "grsr_ground_range_origin": float,
            "grsr_poly_order": int,
            "grsr_zero_doppler_time": "parse_datetime_single",  # single datetime
            "heading": float,
            "incidence_angle_coefficients": "parse_float_vect",
            "incidence_angle_ground_range_origin": float,
            "incidence_angle_poly_order": int,
            "incidence_angle_zero_doppler_time": "parse_datetime_single",  # single datetime
            "incidence_center": float,
            "incidence_far": float,
            "incidence_near": float,
            "look_side": str,
            "mean_earth_radius": float,
            "mean_orbit_altitude": float,
            "number_of_azimuth_samples": int,
            "number_of_dc_estimations": int,
            "number_of_range_samples": int,
            "number_of_state_vectors": int,
            "orbit_absolute_number": int,
            "orbit_direction": str,
            "orbit_processing_level": str,
            "orbit_relative_number": int,
            "orbit_repeat_cycle": int,
            "polarization": str,
            "posX": "parse_float_vect",
            "posY": "parse_float_vect",
            "posZ": "parse_float_vect",
            "processing_prf": float,
            "processing_time": "parse_datetime_single",  # single datetime
            "processor_version": str,
            "product_file": str,
            "product_level": str,
            "product_name": str,
            "product_type": str,
            "range_looks": int,
            "range_sampling_rate": float,
            "range_spacing": float,
            "range_spread_comp_flag": bool,
            "sample_precision": str,
            "satellite_look_angle": str,
            "satellite_name": str,
            "slant_range_to_first_pixel": float,
            "state_vector_time_utc": "parse_datetime_vect",  # 1d vect of datetimes, need to be parsed.
            "total_processed_bandwidth_azimuth": float,
            "velX": "parse_float_vect",
            "velY": "parse_float_vect",
            "velZ": "parse_float_vect",
            "window_function_azimuth": str,
            "window_function_range": str,
            "zerodoppler_end_utc": "parse_datetime_single",  # single datetime
            "zerodoppler_start_utc": "parse_datetime_single",  # single datetime
        }

    elif product_type == "xml":
        raise NotImplementedError
    elif not isinstance(product_type, str):
        raise TypeError(
            'Did not understand input "product_type", a str was expected but a %s datatype variable was input.'
            % type(product_type)
        )
    else:
        raise ValueError(
            'Did not understand input "product_type", either "SLC", "GRD" or "xml" was expected but %s was input.'
            % product_type
        )

    return expected_dtypes


def _fix_GRD_metadata_datatypes(metadata, expected_dtypes):
    """
    Attempt to convert all the metadata fields according to the formula specified
    in expected_dtypes.
    """

    def __parse_float_vect(str_of_vect):
        """
        The 1D vectors are interpreted as strings by the rasterio.keys() reader.
        This aux function splits them into numpy arrays of floating points.
        """

        num_left_brackets = str_of_vect.count("[")
        num_right_brackets = str_of_vect.count("[")

        assert (
            num_left_brackets == num_right_brackets
        ), 'The input was expected to be a str representation of a python list of floats. The number of left brackets "[" and right brackets "]" did not match. The parser will most likely break, aborting.'

        str_of_vect = str_of_vect[1:-1]  # starts and ends with a bracket. Remove them.

        if num_left_brackets == 1:  # single list

            str_of_vect = str_of_vect.replace(",", "")  # remove dots
            str_of_vect = str_of_vect.split(" ")
            while "" in str_of_vect:
                str_of_vect.remove("")

            floats = []
            for i in range(0, len(str_of_vect)):
                floats.append(float(str_of_vect[i]))
            floats = np.array(floats)

        elif num_left_brackets == 0:
            raise ValueError(
                "The input was expected to be a str representation of a python list of floats, but no brackets were found in the input str. The parser will most likely break, aborting."
            )
        else:  # num_left_brackets > 1:
            raise ValueError(
                'The input was expected to be a str representation of a python list of floats, but %d left brackets "[" were found in the input str. The parser will most likely break, aborting.'
                % num_left_brackets
            )

        # numpy array of floats
        return floats

    def __parse_datetime_single(str_of_single):
        """
        Just a single datetime value. Turn it into a numpy array and return.
        """
        return np.array(str_of_single)

    def __parse_datetime_vect(str_of_vect):
        """
        The datetime vectors are interpreted as strings by the rasterio.keys() reader.
        This aux function splits them into lists of strings, which are further parsed in Zerodoppler.py.
        """

        num_left_brackets = str_of_vect.count("[")
        num_right_brackets = str_of_vect.count("]")

        assert (
            num_left_brackets == num_right_brackets
        ), 'The input was expected to be a str representation of a python list of str dates. The number of left brackets "[" and right brackets "]" did not match. The parser will most likely break, aborting.'

        str_of_vect = str_of_vect[1:-1]  # starts and ends with a bracket. Remove them.

        if num_left_brackets == 1:

            str_of_vect = str_of_vect.replace("'", "")
            str_of_vect = str_of_vect.replace(" ", "")
            vect_of_str = str_of_vect.split(",")
            while "" in vect_of_str:
                vect_of_str.remove("")
            vect_of_str = np.array(vect_of_str)

        elif num_left_brackets == 0:
            raise ValueError(
                "The input was expected to be a str representation of a python list of floats, but no brackets were found in the input str. The parser will most likely break, aborting."
            )
        else:  # num_left_brackets > 1:
            raise ValueError(
                'The input was expected to be a str representation of a python list of floats, but %d left brackets "[" were found in the input str. The parser will most likely break, aborting.'
                % num_left_brackets
            )

        # numpy array of chars
        return vect_of_str

    # Main loop. Go through each field in the metadata and parse the contents.
    for key in metadata.keys():
        if key in expected_dtypes:

            var_type = expected_dtypes[key]
            old_val = metadata[key]

            if type(var_type) is type:  # the field specifies a datatype
                new_val = np.array(
                    var_type(old_val)
                )  # Everything is wrapped in a numpy array so that they index like a hdf5 dataset and everything works off the shelf.
            elif type(var_type) is str:
                if var_type == "parse_float_vect":
                    if (
                        key == "dc_estimate_coeffs"
                    ):  # exponential representation, value is truncated too much to trust
                        new_val = None
                    else:
                        new_val = __parse_float_vect(old_val)
                elif var_type == "parse_datetime_vect":
                    new_val = __parse_datetime_vect(old_val)
                elif var_type == "parse_datetime_single":
                    new_val = __parse_datetime_single(old_val)
                else:
                    raise TypeError(
                        'An unsupported var_type "%s" was input.' % var_type
                    )
            else:
                raise TypeError(
                    'Expected the field of expected dtypes to be either a <class "type"> or a str but a %s datatype variable was input.'
                    % type(var_type)
                )

            metadata[key] = new_val

    return metadata


def _raise_rasterio_IOError(path):
    """
    Aux subfunction. Return different error if file was not found than if it just failed to load.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError("No file named %s was found, aborting" % path)
    else:
        raise OSError(
            "Found a file in %s but failed to load it with rasterio.open(), aborting"
            % path
        )


def _parse_GRD_RPC(metadata, source_file):
    """
    Attempt to load the coefficients for a rational polynomial cubic model from the source geotiff metadata,
    as read by coregister.IO.load_ICEYE_metadata()
    """

    RPC_raw = source_file.tags(ns="RPC")

    if not RPC_raw:  # empty dict

        RPC = None
        RPC_metadata = None

    else:

        az_num = _parse_GRD_RPC_vect(RPC_raw["LINE_NUM_COEFF"])
        az_denom = _parse_GRD_RPC_vect(RPC_raw["LINE_DEN_COEFF"])
        range_num = _parse_GRD_RPC_vect(RPC_raw["SAMP_NUM_COEFF"])
        range_denom = _parse_GRD_RPC_vect(RPC_raw["SAMP_DEN_COEFF"])

        if (
            az_num is None
            or az_denom is None
            or range_num is None
            or range_denom is None
        ):
            RPC = None
            RPC_metadata = None
        else:
            RPC = np.zeros((4, 20))

            RPC[0, :] = az_num
            RPC[1, :] = az_denom
            RPC[2, :] = range_num
            RPC[3, :] = range_denom

            RPC_metadata = {
                "lat_mean": float(RPC_raw["LAT_OFF"]),
                "lat_scale_factor": float(RPC_raw["LAT_SCALE"]),
                "lon_mean": float(RPC_raw["LONG_OFF"]),
                "lon_scale_factor": float(RPC_raw["LONG_SCALE"]),
                "height_mean": float(RPC_raw["HEIGHT_OFF"]),
                "height_scale_factor": float(RPC_raw["HEIGHT_SCALE"]),
                "az_idx_mean": float(RPC_raw["LINE_OFF"]),
                "az_idx_scale_factor": float(RPC_raw["LINE_SCALE"]),
                "range_idx_mean": float(RPC_raw["SAMP_OFF"]),
                "range_idx_scale_factor": float(RPC_raw["SAMP_SCALE"]),
            }

    metadata["RPC"] = RPC
    metadata["RPC_metadata"] = RPC_metadata

    return metadata


def _parse_GRD_RPC_vect(str_coeffs):
    """
    The GRD RPC is packed in string lists of 20 elements. Unravel them.
    """

    coeffs = str_coeffs.split(" ")

    if len(coeffs) != 20:
        warnings.warn(
            "Expected there to be 20 rational polyonomial cubic coefficients in the metadata, but %d coefficients were detected. The RPC metadata is corrupted and will be removed."
            % len(coeffs),
            RuntimeWarning,
        )
        coeffs = None
    else:
        coeffs = [float(coeff) for coeff in coeffs]

    return coeffs
