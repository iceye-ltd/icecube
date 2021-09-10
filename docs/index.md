# ICEcube: Python Library for AI-Oriented SAR Datacubes

<a href='http://www.iceye.com'>ICEYE</a> was the first commerical satellite company to miniaturize SAR technology. We have since launched 14 satellites (as of 2021) and continue to grow this constellation. But of course, a large constellation means that we have a lot of data to handle and explore. SAR can be complicated with processing overhead, as the radar phenomenology can require specific expertise.

From an artificial intelligence (AI) and machine learning (ML) perspective, data scientists and engineers have barely scratched the surface with SAR-based implementations. Through a recent initiative with the [European Space Agency (ESA) Φ-lab](https://philab.phi.esa.int/), ICEYE has been able to explore tool development to expedite SAR imagery exploration. 

We were presented with this opportunity to begin creating open source utilities to reduce the barrier to entry into SAR-based machine learning applications. As we are keen to build AI/ML applications to handle heavy image processing and scalable analytics, we have focused our first release on creating a SAR datacube structure to handle time-series SAR as a function of range, azimuth, intensity, and complex radar signal.

If you are a data scientist, ML engineer, or developer who wants to manipulate coregistered stacks of ICEYE SAR images in Python without the burden of handling I/O and memory issues, this library is for you!

We couldn't find tools that enable complex manipulation of high-fidelity SAR images in Python. So, we built one to help you iterate faster. We abstracted out image and metadata formatting, so you can focus on image processing instead of metadata manipulation. 

Feel free to mold this library to suit your needs. You can also use it as a building block in other libraries if you find yourself needing a straightforward tool for I/O and stacking operations. If you don't find it straightforward, we'd love to hear all about it!

## What is a datacube?

A datacube is simply a matrix array of data in three or more dimensions. For us, that is range, azimuth, and time. Once you include amplitude and complex phase information, you are well beyond ‘cube’ into ‘tesseract’ territory, but we will continue to refer to it as a simple cube for the sake of our limited human mind. 

In Figure 1, the satellite is moving in the azimuth direction. Amplitude and phase information is captured at the same point on Earth at different times. The resultant acquisition stack is aligned (coregistered) and stored in ICEcube’s memory-optimized xarray data structures.

<p align="center">
<img src="https://raw.githubusercontent.com/iceye-ltd/icecube/main/assets/datacube_marketing.png?token=ACIXOQMPHXLZ4LQCY4VKRM3BEZQVK" width="491"/>
</p>

<i> Figure 1. The ICEcube tool facilitates the stacking of ICEYE SAR in phase and amplitude as xarray datacubes. ICEYE acquires images at t = 0 for a place on Earth. Subsequent revisits of that point might result in further acquisitions (e.g. t = 1, t = 2, … t = n). The revisited location produces a stack of images that can be aligned (coregistered). The xarray matrix handling is highly memory optimized and facilitates the growth and manipulation of increasingly larger stacks.</i> 

To use ICEcube, all that is required is a stack of ICEYE standard data formats (either ground range detected or single look complex) and a config.json file with information about the date range, min and max incidence angles, and temporal resolution. 

From an ML perspective, the temporal resolution is incredibly valuable for change analysis and quantifying ‘normal’ patterns of life on Earth at specific locations. Preserving this information becomes one of the most critical contributions from the SAR image stack. 

We recognize that to make the most of this time-series cube structure, each subsequent image must be well-aligned (coregistered) with one-another. Despite georegistration of the imagery, often subtle spatial shifts from image-to-image manifest. The act of coregistering these images  is a challenge in itself, so we will be releasing soon a simple coregistration SNAP graph to help you get started.

As an aside, if you are a researcher or an application developer, you can <a href='https://www.iceye.com/free-data/iceye-data-for-research'>submit a proposal</a> to ESA to gain ESA-sponsored (free) ICEYE images for research and pre-operational application development.


**Let's go build your first ICEcube!**

