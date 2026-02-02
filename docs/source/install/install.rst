.. _install_all:

Installation
===================

If you have  **Python 3.11** installed you can setup ``pyvale`` using your preferred package manager:

.. tab-set::
   
   .. tab-item:: pip

      .. code-block:: bash

          # Create a virtual environment with Python 3.11
          python3.11 -m venv venv-pyvale

          # Activate the environment
          source venv-pyvale/bin/activate  # On Windows: .\venv-pyvale\Scripts\Activate.ps1

          # Install pyvale
          pip install pyvale

   .. tab-item:: uv

      .. code-block:: bash

          # Initialize a new project with Python 3.11
          uv init --python 3.11 try-pyvale
          cd try-pyvale

          # Add pyvale as a dependency
          uv add pyvale


   .. tab-item:: conda

      .. code-block:: bash

          # Create a conda environment with Python 3.11
          conda create -n pyvale-env python=3.11

          # Activate the environment
          conda activate pyvale-env

          # Install pyvale
          pip install pyvale

**We have detailed install guides for non-specialist python users for the common operating systems below. 
This includes walkthroughs on how to install the correct python version for your operating system and how to setup a virtual environment.**

.. toctree::
   :maxdepth: 1

   install_linux.rst
   install_mac.rst
   install_windows.rst
