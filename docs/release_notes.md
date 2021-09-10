# version 1.1.0

Major changes in the second release `1.1.0` include: 

- support added to create datacubes with any SAR product (as long as rasterio I/O supported) with (no to) limited cube configuration

- time coordinates added to `xr.Dataset` of labels datacube   

- datacube can be created by providing labels for subset of stack; previously for each raster, a label had to be specified

- changes related to documentation include :
    -  notebooks overview details added   
    -  GRD and SLC notebooks merged into a single SARDatacube notebook  
    -  Ex3-CreatingDatacube added with details on CreatingDatacubes and CubeConfiguration

# version 1.0.0

This is the first public release of the github repository that includes

- creating datacubes with ICEYE L1 products (GRD/SLC)

- creating datacubes with labels (raster/vector)

- configuring datacubes using a JSON file for time, incidence angle and temporal resolution

- dask support to create massive datacubes

- extensive mkdocs documentation to get you easily started with the repository