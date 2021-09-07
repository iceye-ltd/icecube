# Overview

The notebook provides an overview for the notebook examples that are placed under `docs/examples/`. All jupyter notebooks can be viewed online using the navigation link [documentation/ICEcube Examples](https://iceye-ltd.github.io/icecube/examples). 

The notebooks are tailored to demonstrate to users how to fully utilize the benefits of the `icecube` toolkit. We hope that by end of all the examples, you understand how to create datacubes with SAR data and training labels, and use them for machine learning.


We have included following notebooks so far:

- **[Ex1_SARDatacube](https://iceye-ltd.github.io/icecube/examples/Ex1_SARDatacube)** demonstrates how datacubes can be created using ICEYE SAR products (GRD/SLC)

- **[Ex2_LabelsDatacube](https://iceye-ltd.github.io/icecube/examples/Ex2_LabelsDatacube)** shows creating datacubes with ML annotations (raster/vector). This makese  ICEcubes* wholesome for machine learning where input data (X) and output (y) can be found under one umbrella. 

- **[Ex3_CreatingDatacube](https://iceye-ltd.github.io/icecube/examples/Ex3_CreatingDatacube)** walks user how to create datacubes using different methods available.  

- **[Ex4_Datacube](https://iceye-ltd.github.io/icecube/examples/Ex4_Datacube)** briefly describes how to use the `Datacube` class to easily read and process ICEYE datacubes

- **[Ex5_Datacube_ML](https://iceye-ltd.github.io/icecube/examples/Ex5_Datacube_for_ML)** demonstrates a machine learning example with PyTorch using ICEcubes 

*For the sake of clarity, the term `icecube` refers to the toolkit and ICEcube refers to a datacube generated using ICEYE SAR data with the toolkit

## Architecture Diagram

In order to provide you with a better understanding of how different components of `icecube` toolkit communicate with each other, we have included an architecture diagram as shown in Figure 1.

<p align="center">
<img src="https://raw.githubusercontent.com/iceye-ltd/icecube/main/assets/icecube_architecture_diagram.png?token=ACIXOQMPHXLZ4LQCY4VKRM3BEZQVK" />
</p>

<i> Figure 1: Illustrates the architecture diagram of the icecube toolkit. An OOP-oriented architecture ensures a modular approach for the Python library. Low-level details are abstracted away from the users by “IceyeProcessGenerateCube” class. This makes it very easier for users to create cubes without worrying about the implementation details </i>

Figure 1 above provides a holistic overview of the implementation  architecture of the icecube toolkit. The primary components of architecture diagram are briefly described as following:


- **local ICEYE images** denotes a local directory containing coregistered ICEYE stack.
- **user_config.json** contains user-specified configuration for datacube.
- **Labels.json** contains labels for the ICEYE stack in the icecube formatted  JSON structure. 
- **IceyeProcessGenerateCube** is the main class that users interact with. It contains the logic to trigger the right classes (or code blocks).
- **SARDatacubeMetadata** builds metadata from the specified stack of images. This is an efficient way to create a datacube without having to read images in memory.
- **SARDatacube** is the parent class to GRDDatacube and SLCDatacube classes that generates datacube for GRD and SLC images respectively.
- **LabelsDatacube** is the parent class to RasterLabels and VectorLabels that generates datacube  for raster and vector labels respectively. 


Now that you know how icecube toolkit architecture looks like, let's cotinue to examples and build our first ICEcube :)