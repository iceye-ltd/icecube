#!/usr/bin/env python
"""
Global variables used for cube generation and processing.
Please don't configure these unless deem necessary.
"""
import numpy as np

# Cube Coordinates
NAME_RANGE = "Range"
NAME_AZIMUTH = "Azimuth"
NAME_BAND = "Band"

# Cube Data Variables
NAME_INTENSITY_BAND = "Intensity"
NAME_AMPLITUDE_BAND = "Amplitude"
NAME_REAL_BAND = "Real"
NAME_COMPLEX_BAND = "Complex"
NAME_PHASE_BAND = "Phase"

# FOR LABELS DATACUBES
NAME_LABELS_BAND = "Labels"
MASK_DTYPE = "default"
RASTER_DTYPE = "default"

# Config. Variables
DEFAULT_FILL_VALUE = np.nan  # can be very well replaced with 0
CHUNK_SIZE = (4000, 2000)
