#!/usr/bin/env python
"""
Description: Script contains useful utilities for labels cube
"""
import json


def read_json(json_fpath):
    with open(json_fpath) as json_file:
        return json.load(json_file)


def get_labels_type(labels_fpath):
    """
    confirm what is the type of labels : raster or vector
    :param labels_fpath: path/to/labels.json containing labels
    returns "raster" or "vector"
    """
    json_labels = read_json(labels_fpath)
    raster_type, vector_type = False, False

    for _, raster_label in enumerate(json_labels):
        if list(raster_label["labels"].keys())[0] in [
            "segmentation",
            "Segmentation",
            "SEGMENTATION",
        ]:
            raster_type = True

        elif list(raster_label["labels"].keys())[0] in [
            "objects",
            "Objects",
            "OBJECTS",
        ]:
            vector_type = True
        else:
            raise ValueError(
                "Could not understood the imported labels format. Kindly check the format."
            )

        if raster_type and vector_type:
            raise ValueError(
                "Cannot ingest segmentation masks and vector labels at the same time"
            )

    if raster_type:
        return "raster"
    elif vector_type:
        return "vector"
    else:
        raise ValueError("Unknow condition occurred while detecting labels type")
