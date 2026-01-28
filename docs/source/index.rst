.. pyvale documentation master file, created by
   sphinx-quickstart on Wed Apr  2 13:15:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ./_static/pyvale_logo.png
   :alt: Pyvale Logo
   :align: center
   :width: 600px

.. grid:: Pyvale
   :text-align: center
   :margin: 0 
   :shadow: none


Pyvale: The Python Validation Engine
====================================

.. |pypi| image:: https://img.shields.io/pypi/v/pyvale
   :target: https://pypi.org/project/pyvale/
   :alt: PyPI version

.. |github| image:: https://img.shields.io/badge/github-repo-blue?logo=github
   :target: https://github.com/Computer-Aided-Validation-Laboratory/pyvale
   :alt: GitHub repository

.. |wheels| image:: https://img.shields.io/github/actions/workflow/status/Computer-Aided-Validation-Laboratory/pyvale/wheels.yml?branch=main&label=Build
   :target: https://github.com/Computer-Aided-Validation-Laboratory/pyvale/actions/workflows/wheels.yml
   :alt: Wheels Build

.. |tests| image:: https://img.shields.io/github/actions/workflow/status/Computer-Aided-Validation-Laboratory/pyvale/tests.yml?branch=main&label=Tests
   :target: https://github.com/Computer-Aided-Validation-Laboratory/pyvale/actions/workflows/tests.yml
   :alt: Run tests

.. |issues| image:: https://img.shields.io/github/issues/Computer-Aided-Validation-Laboratory/pyvale
   :target: https://github.com/Computer-Aided-Validation-Laboratory/pyvale/issues
   :alt: Open GitHub issues

.. |PR| image:: https://img.shields.io/github/issues-pr/Computer-Aided-Validation-Laboratory/pyvale
   :target: https://github.com/Computer-Aided-Validation-Laboratory/pyvale/issues
   :alt: Open GitHub Pull Requests

.. |license| image:: https://img.shields.io/github/license/Computer-Aided-Validation-Laboratory/pyvale
   :target: https://choosealicense.com/licenses/mit/
   :alt: MIT License

|pypi| |github| |wheels| |tests| |issues| |PR| |license|


Pyvale aims to become an all-in-one package for sensor uncertainty quantification simulation, experimental design, sensor placement optimisation and simulation calibration/validation.
Used to simulate experimental data from an input multi-physics simulation by explicitly modelling sensors with realistic uncertainties.
We are actively developing dedicated tools for simulation and uncertainty quantification of imaging sensors including digital image correlation (DIC) and infra-red thermography (IRT).

Getting Started
---------------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: User Guide 
      :link: guide_user/guide_user
      :link-type: doc
      :text-align: center
      :shadow: lg
      
      :octicon:`workflow;5em`

   .. grid-item-card:: Theory Overview
      :link: guide_theory/guide_theory
      :link-type: doc
      :text-align: center
      :shadow: lg
      
      :octicon:`book;5em`

   .. grid-item-card:: Installation
      :link: install/install
      :link-type: doc
      :text-align: center
      :shadow: lg

      :octicon:`gear;5em`

   .. grid-item-card:: Examples
      :link: examples/examples
      :link-type: doc
      :text-align: center
      :shadow: lg

      :octicon:`file-code;5em`


.. toctree::
    :hidden:

    guide_user/guide_user
    guide_theory/guide_theory
    install/install
    examples/examples
    api_py
    api_cpp
    cite


Citing Pyvale
---------------

If you use the code in your published work, then please cite the following article:

.. tab-set::

   .. tab-item:: MLA

         Hirst, Joel, et al. "PYVALE: A Fast, Scalable, Open-Source 2D Digital Image Correlation 
         (DIC) Engine Capable of Handling Gigapixel Images." 
         *arXiv preprint arXiv:2601.12941* (2026).

   .. tab-item:: APA

         Hirst, J., Sibson, L., Tayeb, A., Poole, B., Sampson, M., Bielajewa, W., ... & Fletcher, L. (2026). 
         PYVALE: A Fast, Scalable, Open-Source 2D Digital Image Correlation (DIC) Engine 
         Capable of Handling Gigapixel Images. 
         *arXiv preprint arXiv:2601.12941*.

   .. tab-item:: Bibtex

      .. code-block::

         @article{pyvale2026,
            title={PYVALE: A Fast, Scalable, Open-Source 2D Digital Image Correlation (DIC) Engine Capable of Handling Gigapixel Images},
            author={Hirst, Joel and Sibson, Lorna and Tayeb, Adel and Poole, Ben and Sampson, Megan and Bielajewa, Wiera and Atkinson, Michael and Marsh, Alex and Spencer, Rory and Hamill, Rob and others},
            journal={arXiv preprint arXiv:2601.12941},
            year={2026}
        }


Key Contributors
-----------------

The Computer Aided Validation Team at United Kingdom Atomic Energy Authority (UKAEA):

* Lloyd Fletcher (`ScepticalRabbit <https://github.com/ScepticalRabbit>`_), UK Atomic Energy Authority
* Joel Hirst (`JoelPhys <https://github.com/JoelPhys>`_), UK Atomic Energy Authority
* Lorna Sibson (`lornasibson <https://github.com/lornasibsin>`_), UK Atomic Energy Authority
* Megan Sampson (`megan sampson <https://github.com/meganasampson>`_), UK Atomic Energy Authority
* Adel Tayeb (`3adelTayeb <https://github.com/3adelTayeb>`_), UK Atomic Energy Authority
* Alex Marsh (`alexmarsh2 <https://github.com/alexmarsh2>`_), UK Atomic Energy Authority
* Rory Spencer (`fusmatr <https://github.com/fusmatrs>`_), UK Atomic Energy Authority
* John Charlton  (`coolmule0 <https://github.com/coolmule0>`_), UK Atomic Energy Authority




