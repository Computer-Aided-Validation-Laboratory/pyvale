.. _install_all:

Installation
===================

If you have **Python 3.11 or newer** installed, you can set up ``pyvale``
using your preferred package manager. Blender is optional and currently
requires Python 3.13.

.. tab-set::
   
   .. tab-item:: pip

      .. code-block:: bash

          # Create a virtual environment with a supported Python version
          python -m venv venv-pyvale

          # Activate the environment
          source venv-pyvale/bin/activate  # On Windows: .\venv-pyvale\Scripts\Activate.ps1

          # Install pyvale
          pip install pyvale

   .. tab-item:: uv

      .. code-block:: bash

          # Initialize a new project with a supported Python version
          uv init try-pyvale
          cd try-pyvale

          # Add pyvale as a dependency
          uv add pyvale


   .. tab-item:: conda

      .. code-block:: bash

          # Create a conda environment with Python 3.13
          conda create -n pyvale-env python=3.11

          # Activate the environment
          conda activate pyvale-env

          # Install pyvale
          pip install pyvale

**We have detailed install guides for non-specialist python users for the common operating systems below. 
This includes walkthroughs on how to install the correct python version for your operating system and how to setup a virtual environment.**

Blender backend
----------------

The default rendering backend is Riley. Blender is an optional backend and is
available only with Python 3.13:

.. code-block:: bash

   pip install "pyvale[blender]"

.. toctree::
   :maxdepth: 1

   install_linux.rst
   install_mac.rst
   install_windows.rst
