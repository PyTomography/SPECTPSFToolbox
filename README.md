# SPECT Point Spread Function Toolbox

This toolbox provides functionality for developing and fitting PSF models to SPECT point source data; the developed PSF models can be loaded into [PyTomography](https://github.com/qurit/PyTomography) for general PSF modeling. For more information, including installation instructions and tutorials, please see the [documentation website](https://spectpsftoolbox.readthedocs.io/en/latest/). If you wish to contribute, please see the `CONTRIBUTING.md` file.

## Context

Gamma cameras used in SPECT imaging have finite resolution: infinitesmal point sources of radioactivity show up as finite "point spread functions" (PSF) on the camera. Sample PSFs from point sources at various distances from a camera can be seen on the left hand side of the figure below.

 The PSF consists of three main components: (i) the geometric component (GC) which depends on the shape and spacing of the collimator bores, (ii) the septal penetration component (SPC) which results from photons that travel through the collimator material without being attenuated, and (iii) the septal scatter component (SSC), which consists of photons that scatter within the collimator material and subsequently get detected in the scintillator. When the thickness of the SPECT collimator sufficiently matches the energy of the detected radiation, the PSF is dominated by the GC and can be sufficiently approximated using a distance dependent Gaussian function. When the energy of the photons is large relative to the thickness and hole size of the collimator material, the PSF contains significant contributions from SPC and SSC and it can no longer be approximated using simple Gaussian functions. For more information, see [chapter 16 of the greatest book of all time](https://www.wiley.com/en-in/Foundations+of+Image+Science-p-9780471153009)

 The figure below shows axial slices of reconstructed Monte Carlo Ac225 SPECT data. The images highlighted as "PSF Model" correspond to application of a PSF operator developed in the library on a point source. The images highlighted as "New" are obtainable via reconstruction with [PyTomography](https://github.com/qurit/PyTomography) using the PSF operators obtained in this library; they require comprehensive PSF modeling. 

 ![ac225](figures/ac_recon_modeling.png "Ac225 PSF Modeling and Reconstruction")



