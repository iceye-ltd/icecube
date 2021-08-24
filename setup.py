import os

from setuptools import setup, find_packages


def read(fname):
    """
    # Utility function to read the README file for long descriptions.
    :param fname:
    :return:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements = [
    "numpy",
    "rasterio",
    "h5py",
    "xmltodict",
    "tqdm",
    "pip",
    "scikit-image",
    "opencv-python",
    "shapely",
    "pandas",
    "xarray",
    "dask",
    "netCDF4",
    "bottleneck",
]

dev_requirements = [
    "invoke",
    "pre-commit",
    "black",
    "flake8",
    "black",
    "bump2version",
    "pytest",
    "mkdocs>=1.1.2",
    "mkdocs-material>=7.1.4",
    "mkdocs-jupyter>=0.17.3",
    "mknotebooks>=0.6.2",
    "mkdocstrings>=0.3.2",
]

setup(
    name="icecube",
    version="1.0.0beta1",
    author="Muhammad Irfan Ali, Arnaud Dupeyrat",
    author_email="irfan.ali@iceye.com, arnaud.dupeyrat@iceye.fi",
    description="AI oriented datacube generation using ICEYE data",
    license="Copyright Iceye Oy",
    keywords="data cubes, ML cubes, iceye cubes",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    long_description=read("Readme.md"),
    project_urls={
        "Bug Tracker": "",
        "Documentation": "",
        "Source Code": "https://github.com/iceye-ltd/icecube",
    },
    entry_points={
        'console_scripts': ['icecube = icecube.bin.generate_cube:cli'],
    },
)
