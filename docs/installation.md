# Installation

## Before you begin

You need Python 3.8 or later to use the ICEcube library. 

The installation options depend on whether you want to use the library in your Python scripts or you want to contribute to it.


## To use the library
We recommend that you create a Conda environment to keep your projects organized. 

Perform the following steps to install the ICEcube library.

``` 
git clone https://github.com/iceye-ltd/icecube.git 
cd icecube
conda env create -f environment.yml
conda activate icecube_env 
conda install -c conda-forge xarray dask netCDF4 bottleneck
pip install -e .  
```

<br>

## To contribute to the library

Clone the repository and run the following commands to install the required dependencies. For more information, see the `Makefile`.

```
pip install invoke
inv setup
```

>**Note:** A useful command is `inv -l `

For information on contributing to the library, see the [Contributing](contribute.md) page.