.. SPECTPSFToolbox documentation master file, created by
   sphinx-quickstart on Fri Aug  2 21:56:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPECTPSFToolbox Documentation
===========================================

The ``SPECTPSFToolbox`` is a collection of python functions and classes that can be used for modeling the point spread functions (PSFs) of gamma cameras. The PSF models developed by the toolbox can be saved and used in `PyTomography <https://pytomography.readthedocs.io/en/latest/>`_ for SPECT image reconstruction.

In traditional SPECT reconstruction, only the geometric component of the PSF is considered; this library provides a means for modeling the septal scatter and septal penetration components, permitting more accurate image reconstruction when these components are significant.

.. grid:: 1 3 3 3
    :gutter: 2
    
    .. grid-item-card:: Tutorials
        :link: tutorials/tutorials
        :link-type: doc
        :link-alt: Tutorials
        :text-align: center

        :material-outlined:`psychology;8em;sd-text-secondary`

        These tutorials show how to construct operators, fit them to real/Monte Carlo PSF data, and use them in image reconstruction.

    .. grid-item-card:: API Reference
        :link: autoapi/index
        :link-type: doc
        :link-alt: API
        :text-align: center

        :material-outlined:`computer;8em;sd-text-secondary`

        View the application programming interface of the library

    .. grid-item-card:: Get Help
        :text-align: center

        :material-outlined:`live_help;8em;sd-text-secondary`

        .. button-link:: https://github.com/lukepolson/SPECTPSFToolbox/issues
            :shadow:
            :expand:
            :color: warning

            **Report an issue**

        .. button-link:: https://pytomography.discourse.group/
            :shadow:
            :expand:
            :color: warning

            **Ask questions on PyTomography discourse**

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials/tutorials