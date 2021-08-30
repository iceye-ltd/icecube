#!/usr/bin/env python
"""
Description: The script provides helpful functions to create labels in a format compliant to datacubes.
"""
import json


class CreateLabels:
    def __init__(self, labels_type):
        self.labels_type = labels_type
        self._validate_labels_type()
        self.labels_collection = []
        self.initate_labels_mapping()

    def _validate_labels_type(self):
        self.known_types = ["vector", "raster"]

        if self.labels_type not in self.known_types:
            raise ValueError(
                f"Labels type must be one of the folliwng: {self.known_types}"
            )

    def initate_labels_mapping(self):
        self.creat_vector_labels = lambda p_file, p_label: {
            "product_file": p_file,
            "labels": {"objects": p_label},
        }
        self.creat_raster_labels = lambda p_file, mask_fpath: {
            "product_file": p_file,
            "labels": {"segmentation": mask_fpath},
        }

    def populate_labels(self, product_file: str, product_labels):
        assert type(product_file) == str, "product_file must be of type str"
        if self.labels_type == "vector":
            self.populate_vector_labels(product_file, product_labels)
        elif self.labels_type == "raster":
            self.populate_raster_labels(product_file, product_labels)
        else:
            raise ValueError("Unknow condition occurred")

    def populate_vector_labels(self, product_file: str, product_labels):
        assert (
            type(product_labels) == dict or type(product_labels) == list
        ), "product_labels must be dict/list for vector type"

        if self.product_file_exists(product_file):
            working_index = self.get_product_file_index(product_file)
            self.labels_collection[working_index]["labels"]["objects"].append(
                product_labels
            )
        else:
            if type(product_labels) == dict:
                product_labels = list(product_labels)
            self.labels_collection.append(
                self.creat_vector_labels(product_file, product_labels)
            )

    def populate_raster_labels(self, product_file: str, mask_fpath: str):
        assert type(mask_fpath) == str, "product_labels must be string for raster type"
        if self.product_file_exists(product_file):
            raise ValueError("Cannot append multiple masks against the same raster")
        self.labels_collection.append(
            self.creat_raster_labels(product_file, mask_fpath)
        )

    def product_file_exists(self, product_file):
        for indx, row_dict in enumerate(self.labels_collection):
            if product_file == row_dict["product_file"]:
                return True
            continue

        return False

    def get_product_file_index(self, product_file):
        for indx, row_dict in enumerate(self.labels_collection):
            if (
                product_file == row_dict["product_file"]
                and self.labels_type == "raster"
            ):
                raise ValueError(
                    "Cannot append multiple masks against the same product_file"
                )
            elif product_file == row_dict["product_file"]:
                return indx

        return len(self.labels_collection) - 1

    def write_labels_to_json(self, out_fpath, ensure_ascii=True):
        try:
            with open(out_fpath, "w", encoding="utf-8") as f:
                json.dump(
                    self.labels_collection, f, ensure_ascii=ensure_ascii, indent=4
                )

        except ValueError as e:
            print(f"Error: {e} while writing labels to json")

    def create_instance_bbox(self, obj_class: str, bbox_dict: dict):
        return {"class": obj_class, "bbox": bbox_dict}

    def create_instance_polygon(self, obj_class: str, polygon_points: list):
        return {"class": obj_class, "polygon": polygon_points}

    def create_instance_point(self, obj_class: str, points: dict):
        return {"class": obj_class, "point": points}

    def create_instance_classification(self, tags: list):
        return {"classification": tags}

    def create_instance_segmentation(self, mask_fpath: str):
        return str(mask_fpath)
