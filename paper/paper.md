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
    orcid: 0000-0002-3182-2782
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

`SPECTPSFToolbox` is a python toolbox built with PyTorch for point spread function (PSF) modeling in clinical Single photon emission computed tomography (SPECT) imaging. The toolbox provides functions and classes that model distinct components of SPECT PSFs for parallel hole collimator systems. The individual components can be chained through addition and multiplication to construct a full PSF model. Developed models may also contain parameters which can be fit to Monte Carlo (MC) or real SPECT projection data. The toolbox is an independent repository of the PyTomography [@pytomography] project; developed models can be directly used in the main PyTomography library for SPECT image reconstruction. 


# Statement of need

SPECT is an \textit{in vivo} imaging modality modality used to estimate the 3D radiopharmaceutical distributions within a patient [@Zaidi2006]. It requires (i) acquisition of 2D ``projection'' images at different angles using a gamma camera followed by (ii) use of a tomographic image reconstruction algorithm to obtain a 3D radioactivity distribution consistent with the acquired data. In order to reconstruct SPECT projection data, a system model is needed that captures all features of the imaging system, such as resolution modeling. The gamma cameras used in SPECT imaging have finite resolution, meaning point sources of radioactivity appear as point spread functions (PSFs) on the camera. The PSF has two main components: (i) the intrisic response function (IRF), which results from the inability of the scintillator to precisely locate the point of interaction and (ii) the collimator detector response function (CDRF), which results from the inability of the collimator to select for photons travelling perpendicular to the bores.

The CDRF itself consists of three main components: (i) the geometric response function (GRF) [@metz1980geometric; @GRF_2;@GRF_3; @GRF_4] which results from photons that pass through the collimator bores without intersecting the septa, (ii) the septal penetration response function (SPRF) which results from photons that travel through the septa without being attenuated, and (iii) the septal scatter response function (SSRF), which consists of photons that scatter within the collimator material and subsequently get detected in the scintillator [@septal]. As the thickness of the collimator increases and diameter of the collimator bores decreases, the relative contribution from the SPRF and SSRF decreases. When the SPRF and SSRF are sufficiently small, the net PSF is dominated by the IRF and GRF and it can be reasonably approximated using a Gaussian function. The trade-off, however, is that a thick collimator with narrow bores also has lower detector sensitivity. It may necessary to have non-negligible contributions from the SPRF and SSRF in order to increase detector sensitivity. In certain situations, the imaged photons have energies so high that the available commerical collimators are unable to surpress the SPRF and SSRF.

Unfortunately, the exisiting open source reconstruction libraries only provide support for Gaussian PSF modeling and thus can only be used reconstruct SPECT data where the PSF is dominated the IRF and GRF. In many recent SPECT applications, this does not hold. For example, ${}^{225}$Ac based treatments have recently shown promise in clinical studies [@ac1; @ac2; @ac3; @ac4; @ac5; @ac6], and targetted alpha therapy with ${}^{213}$Bi has shown promise in reducing amyloid plaque concentrations in male mice, eluding to a potential treatment option for Alzeimer disease [@bi213_amyloid]. Both ${}^{225}$Ac and ${}^{213}$Bi are imaged using a 440 keV photon emissions that result in significant SPRF and SSRF components in the PSF. If the nuclear medicine community is to explore and develop novel reconstruction techniques in these domains, then there is a need for open source tools that provide comprehensive and computationally efficient SPECT PSF modeling. This python based toolbox provides those tools.

# Overview of SPECTPSFToolbox

The purpose of SPECT reconstruction is to estimate the 3D radionuclide concentration $f$ that produces the acquired 2D image data $g$ given an analytical model for the imaging system, known as the system matrix. Under standard conditions, the SPECT system matrix estimates the projection $g_{\theta}$ at angle $\theta$ as

\begin{equation}
    g_{\theta}(x,y) = \sum_{d} \mathrm{PSF}(d) \left[f'(x,y,d)\right]
    \label{eq:model_approx}
\end{equation}

where $(x,y)$ is the position on the detector, $d$ is the perpendicular distance to the detector, $f'$ is the attenuation adjusted image corresponding to the detector angle, and $\mathrm{PSF}(d)$ is a 2D linear operator that operates seperately on $f$ at each distance $d$. The toolbox provides the necessary tools to obtain $\mathrm{PSF}(d)$.

The toolbox is seperated into three main class types; they are best described by the implementation of their `__call__` methods:

1. `Kernel1D`: called using 1D positions $x$, source-detector distances $d$, hyperparameters $b$, and return a 1D kernel at each source-detector distance.
2. `Kernel2D`: called using a 2D meshgrid  $(x,y)$, source-detector distances $d$, hyperparameters $b$, and return a 2D kernel at each source-detector distance.
3. `Operator`: called using a 2D meshgrid $(x,y)$, source-detector distances $d$, as well as an input $f$, and return the operation $\mathrm{PSF}(d) \left[f'(x,y,d)\right]$

Various subclasses of `Kernel1D` have their own instantiation methods. For example, the `__init__` method of `FunctionKernel1D` requires a 1D function definition $k(x)$, an amplitude function $A(d,b_A)$ and its hyperparameters $b_A$, and a scaling function $\sigma(d,b_{\sigma})$ and its hyperparameters $b_{\sigma}$. The `__call__` method returns $A(d,b_A)k(x/\sigma(d,b_{\sigma})) \Delta x$ where $\Delta x$ is the spacing of the kernel. The `ArbitraryKernel1D` is similar except that it requires a 1D array $k$ in place of $k(x)$, and $k(x/\sigma(d,b_{\sigma}))$ is obtained via interpolation between array values. Subclasses of `Kernel1D` require a `normalization_constant(x,d)` method to be implemented that returns sum of the kernel from $x=-\infty$ to $x=\infty$ at each detector distance $d$ given $x$ input with constant spacing $\Delta x$. This is not as simple as summing over the kernel output since the range of $x$ provided might be less than the size of the kernel. The `Kernel2D` class and corresponding subclasses are analogous to `Kernel1D`, except they require a 2D input $(x,y)$ and return a corresponding 2D kernel at each detector distance $d$. 

Sublasses of the `Operator` class form the main components of the library, and are built using various `Kernel1D` and `Kernel2D` classes. Currently, the library supports linear shift invariant (LSI) operators, since these are sufficient for SPECT PSF modeling. LSI operators can always be implemented via convolution with a 2D kernel, but often this is computationally expensive. In tutorial 5 on the documentation website, the form of the SPECT PSF is exploited and a 2D LSI operator is built using 1D convolutions and rotations. In tutorial 7, this is shown to lead to faster reconstruction than application of a 2D convolution but with nearly identical results. Operators must also implement a `normalization_constant(xv,yv,d)` method. The `__add__` and `__mult__` methods of operators have been implemented so that multiple PSF operators can be chained together; adding operators together yields an operator that returns a linear operator sum, while multiplication yields the equivalent to the linear operator matrix product. Propagation of normalization is implemented to ensure the chained operator is also properly normalized. For example, if the response functions are implemented in operators with variable names `irf_op`, `grf_op`, `ssrf_op`, and `sprf_op`, then the total response would be defined as `psf_op=irf_op*(grf_op+ssrf_op+sprf_op)` and the `__call__` method would implement

\begin{equation}
    \mathrm{PSF} \left[f \right] = \mathrm{IRF}\left(\mathrm{GRF}+\mathrm{SSRF}+\mathrm{SPRF}\right)\left[f \right]
    \label{eq:psf_tot}
\end{equation}

The library is presently demonstrated on two SPECT image reconstruction examples where the ordered subset expectation maximum (OSEM) [@osem] reconstruction algorithm is used. The first use case considers reconstruction of MC ${}^{177}$Lu data. ${}^{177}$Lu is typically acquired using a medium energy collimator and the PSF is dominated by the GRF. When acquired using a low energy collimator, there is significant SPRF and SSRF, and a more sophisticated PSF model is required during reconstruction. This use case considers a cylindrical phantom with standard NEMA sphere sizes filled at a 10:1 source to background concentration with a total activity of 1000 MBq. SPECT acquisition was simulated in SIMIND [@simind] using (i) low energy collimators and (ii) medium energy collimators with 96 projection angles at 0.48 cm $\times$ 0.48 cm resolution and a 128 $\times$ 128 matrix size. Firstly, each case was reconstructed with only the GRF (Gaussian PSF) with OSEM(4it,8ss) (medium energy) and OSEM(40it8ss) (low energy). A MC based PSF model that encompasses GRF, SPRF, and SSRF was then obtained by (i) simulating a point source at 1100 distances between 0 cm and 55cm, and normalizing the kernel data and (ii) using a `NearestKernelOperator` which convolves the PSF kernel closest to the source-detector distance of each plane. The low energy collimator data was then reconstructed using the MC PSF model using OSEM (40it8ss). \autoref{fig:fig1} shows the sample PSF at six sample distances (left), the reconstructed images (center) and sample 1D profiles of the reconstructed images (right). When the MC PSF kernel that includes GRF+SSRF+SPRF is used, the activity in the spheres is significantly higher than when the GRF only (Gaussian) kernel is used. 

The second use case considers reconstruction of MC ${}^{225}$Ac data. ${}^{225}$Ac emits 440 keV photons that have significant SPRF and SSRF even when a high energy collimator is used. This use case considers a cylindrical phantom with 3 spheres of diameters 60mm, 37mm, and 28mm filled at a 6.4:1 source to background ratio with 100MBq of total activity. 440keV point source data was simulated via SIMIND at 1100 positions between 0cm and 55cm. 12 of these positions were used for developing and fitting a PSF operator built using the `GaussianOperator`, `Rotate1DConvOperator`, and `RotateSeperable2DConvOperator`; each of these classes performs 2D shift invariant convolutions using only 1D convolutions and rotations 
(1D-R). More details on this model and how it is fit are shown in tutorial 5 on the documentation website. The developed model is compared to a MC based model (2D) that uses the `NearestKernelOperator` with the 1100 acquired PSFs acquired from SIMIND. Use of the 1D-R model in image reconstruction reduces the required computation time by more than a factor of two; this occurs since rotations and 1D convolutions are significantly faster than direct 2D convolution, even when fast fourier transform techniques are used.

![Upper: ${}^{177}$Lu reconstruction example. From left to right: MC PSF data at various source detector distances, axial slices from reconstructions, and central vertical 1D profile from shown axial slices. Lower: ${}^{225}$Ac reconstruction example. From left to right: MC PSF data and predicted 1D-R fit, axial slices from reconstructions, and central vertical 1D profile from shown axial slices.\label{fig:fig1}](fig1.png)


# References