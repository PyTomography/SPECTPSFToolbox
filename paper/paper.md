---
title: 'SPECTPSFToolbox: A Python Toolbox for SPECT Point Spread Function Modeling'
tags:
  - 3D slicer
  - nuclear medicine
  - tomography
  - spect
  - image reconstruction
authors:
  - name: Luke Polson
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Carlos Uribe
    affiliation: "2, 4, 5"
  - name: Arman Rahmim
    affiliation: "1, 2, 3, 5"
affiliations:
 - name: Deparment of Physics & Astronomy, University of British Columbia, Vancouver Canada
   index: 1
 - name: Department of Integrative Oncology, BC Cancer Research Institute, Vancouver Canada
   index: 2
 - name: School of Biomedical Engineering, University of British Columbia, Vancouver Canada
   index: 3
 - name: Department of Radiology, University of British Columbia, Vancouver Canada
   index: 4
 - name: Molecular Imaging and Therapy Department, BC Cancer, Vancouver Canada
   index: 5
date: 06 July 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`SPECTPSFToolbox` is a python toolbox for SPECT point spread function (PSF) fitting. The library provides functionality for obtaining comprehensive PSF models that can be used in the python library PyTomography [@pytomography] for SPECT reconstruction. The models are constructed from linear operators that can be chained together and fit to Monte Carlo (MC) or real point source data.


# Statement of need

SPECT imaging is used to measure the 3D distribution of a radioactive molecule within a patient; it requires (i) acquisition of 2D projection images at different angles using a gamma camera followed by (ii) use of a tomographic image reconstruction algorithm to obtain a 3D radioactivity distribution consistent with the acquired data [@Barrett2003FoundationsOI]. The gamma cameras used in SPECT imaging have finite resolution: infinitesmal point sources of radioactivity show up as finite "point spread functions" (PSF) on the camera. The PSF consists of three main components: (i) the geometric component (GC) which depends on the shape and spacing of the collimator bores, (ii) the septal penetration component (SPC) which results from photons that travel through the collimator material without being attenuated, and (iii) the septal scatter component (SSC), which consists of photons that scatter within the collimator material and subsequently get detected in the scintillator. When the thickness of the SPECT collimator sufficiently matches the energy of the detected radiation, the PSF is dominated by the GC and can be sufficiently approximated using a distance dependent Gaussian function. When the energy of the photons is large relative to the thickness and hole size of the collimator material, the PSF contains significant contributions from SPC and SSC and it can no longer be approximated using simple Gaussian functions.

Unfortunately, many currently available open source reconstruction library only provide support for Gaussian PSF modeling which can only capture the GC. Given the recent success of ${}^{225}$Ac based radiopharmaceuticals for prostate cancer treatment that emit high energy photons [@ac1; @ac2; @ac3; @ac4; @ac5; @ac6], there is a need for tools that provides comprehensive SPECT PSF modeling that include SPC and SSC. This python based library provides the tools to build general SPECT PSF models. The developed models can be saved and used in open source image reconstruction libraries, such as PyTomography.


# Overview of SPECTPSFToolbox

The purpose of SPECT reconstruction is to estimate the 3D radionuclide concentration $f$ that produces the acquired detector data $g$ given an analytical model for the imaging system, known as the system matrix. Under standard conditions, the SPECT system matrix estimates the projection $g_{\theta}$ at angle $\theta$ as

\begin{equation}
    g_{\theta}(x,y) = \sum_{d} \mathrm{PSF}(d) \left[f'(x,y,d)\right]
    \label{eq:model_approx}
\end{equation}

where $(x,y)$ is the position on the detector, $d$ is the perpendicular distance to the detector, $f'$ is the attenuation adjusted image corresponding to the detector angle, and $\mathrm{PSF}(d)$ is a 2D linear operator that operates seperately on $f$ at each distance $d$. The toolbox provides the necessary tools to obtain $\mathrm{PSF}(d)$.

The toolbox is seperated into three main classes

\begin{enumerate}
\item `Kernel1D`: objects that take in 1D position $x$, source-detector distance $d$, hyperparameters $b$, and return a 1D kernel at each source-detector distance.
\item `Kernel2D`: objects that take in a 2D meshgrid  $(x,y)$, source-detector distance $d$, hyperparameters $b$, and return a 2D kernel at each source-detector distance.
\item `Operator`: objects take in take in a 2D meshgrid $(x,y)$, source-detector distance $d$, as well as an input $f$, and return the operation $\mathrm{PSF}(d) \left[f'(x,y,d)\right]$
\end{enumerate}

`Kernel1D` objects can be constructed using a variety of functions. For example, the `__init__` method of `FunctionKernel1D` requires a 1D function definition $k(x)$, an amplitude function $A(d,b_A)$ and its hyperparameters $b_A$, and a scaling function $\sigma(d,b_{\sigma})$ and its hyperparameters $b_{\sigma}$. The `__call__` method returns $A(d,b_A)k(x/\sigma(d,b_{\sigma})) \Delta x$ where $\Delta x$ is the spacing of the kernel. `Kernel1D` objects require a `normalization_constant(x,d)` method to be implemented that returns sum of the kernel from $x=-\infty$ to $x=\infty$ at each detector distance $d$ given $x$ input with constant spacing $\Delta x$. This is not as simple as summing over the kernel output since the range of $x$ provided might be less than the size of the kernel. `Kernel2D` objects are analogous to `Kernel1D` objects, except they take in a 2D input $(x,y)$ and return a corresponding 2D kernel at each detector distance $d$. 

`Operator` objects are the main component of the library; specific subclasses of operators can be built using `Kernel1D` and `Kernel2D` objects. Currently, the library has support for linear shift invariant (LSI) operators, since these are nearly always used for SPECT PSF modeling. LSI operators can always be implemented via convolution with a 2D kernel, but often this is computationally expensive. In tutorial 5, the form of the SPECT PSF is exploited and a 2D LSI operator is built using 1D convolutions and rotations. In tutorial 7, this is shown to lead to faster reconstruction than application of a 2D convolution but with nearly the same results. Operators must also implement a `normalization_constant(xv,yv,d)` method. The `__add__` and `__mult__` methods of operators have been implemented so that multiple PSF operators can be chained together; adding operators together yields an operator that returns the sum of the two PSFs applied to the input data, while multiplying operators yields an operator that returns successive operation of the PSFs applied to the input data. Propagation of normalization is implemented to ensure the normalization of the chained operator is properly implemented. In a gamma camera, for example, there are two components of the PSF: (i) blurring induced from the collimator and (ii) blurring induced from the intrisic resolution of the scintillator crystals. Provided each operator is instantiated as `psf_coll` and `psf_scint` respectively, the chained PSF operator is `psf_tot=psf_scint*psf_coll`; the `__call__` method of this operator implements

\begin{equation}
    \mathrm{PSF}_{\mathrm{total}} \left[f \right] = \mathrm{PSF}_{\mathrm{scint}}\left[\mathrm{PSF}_{\mathrm{coll}} \left[f \right] \right]
    \label{eq:psf_tot}
\end{equation}

The library is demonstrated using two use cases via reconstruction with the ordered subset expectation maximum (OSEM) algorithm [@osem]. The first use case considers reconstruction of MC ${}^{177}$Lu data. ${}^{177}$Lu is typically acquired using a medium energy collimator and the PSF is dominated by the GC. When acquired using a low energy collimator, there is significant SPC and SSC, and a more sophisticated PSF model is required during reconstruction. This use case considers a cylindrical phantom with standard NEMA sphere sizes filled at a 10:1 source to background concentration with a total activity of 1000 MBq. SPECT acquisition was simulated in SIMIND [@simind] using (i) low energy collimators and (ii) medium energy collimators with 96 projection angles at 0.48 cm $\times$ 0.48 cm resolution and a 128 $\times$ 128 matrix size. Firstly, each case was reconstructed with GC PSF modeling (Gaussian PSF) with OSEM(4it,8ss) (medium energy) and OSEM(40it8ss) (low energy). A MC based PSF model that encompasses GC, SPC, and SSC was then obtained by (i) simulating a point source at 1100 distances between 0 cm and 55cm, and normalizing the kernel data and (ii) using a `NearestKernelOperator` which convolves the PSF kernel closest to the source-detector distance of each plane. The low energy collimator data was then reconstructed using the MC PSF model using OSEM (40it8ss). \autoref{fig:fig1} shows the sample PSF at six sample distances (left), the reconstructed images (center) and sample 1D profiles of the reconstructed images (right). When the MC PSF kernel that includes GC+SSC+SPC is used, the activity in the spheres is significantly higher than when the GC only (Gaussian) kernel is used. 

The second use case considers reconstruction of MC ${}^{225}$Ac data. ${}^{225}$Ac emits 440keV photons that have significant SPC and SSC even when a high energy collimator is used. This use case considers a cylindrical phantom with 3 spheres of diameters 60mm, 37mm, and 28mm filled at a 6.4:1 source to background ratio with 100MBq of total activity. 440keV point source data was simulated via SIMIND at 1100 positions between 0cm and 55cm. 12 of these positions were used for developing and fitting a PSF operator built using the `GaussianOperator`, `Rotate1DConvOperator`, and `RotateSeperable2DConvOperator`; each of these classes performs 2D shift invariant convolutions using only 1D convolutions and rotations 
(1D-R). More details on this model and how it is fit are shown in tutorial 5 on the GitHub page. The developed model is compared to a MC based model (2D) that uses the `NearestKernelOperator` with the 1100 acquired PSFs acquired from SIMIND. Use of the 1D-R model in image reconstruction reduces the required computation time by a factor of 2.5, since rotations and 1D convolutions are significantly faster than direct 2D convolution, even when fast fourier transform techniques are used.

![Upper: ${}^{177}$Lu reconstruction example. From left to right: MC PSF data at various source detector distances, axial slices from reconstructions, and central vertical 1D profile from shown axial slices. Lower: ${}^{225}$Ac reconstruction example. From left to right: MC PSF data and predicted 1D-R fit, axial slices from reconstructions, and central vertical 1D profile from shown axial slices.\label{fig:fig1}](fig1.png)


# References