Installation
===================

Windows
---------------

Mac OS
------------


Ubuntu linux
-------------

Managing Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~

To be compatible with ``bpy`` (the Blender python interface), ``pyvale`` uses python 3.11. To install python 3.11 without corrupting your operating systems python installation first add the deadsnakes repository to apt:

.. code-block:: bash

   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update && sudo apt upgrade -y

Install python 3.11:

.. code-block:: bash

   sudo apt install python3.11

Add ``venv`` to your python 3.11 install:

.. code-block:: bash

   sudo apt install python3.11-venv

Check your python 3.11 install is working using the following command which should open an interactive python interpreter:

.. code-block:: bash

   python3.11

Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing ``pyvale`` in a virtual environment using ``venv`` or ``pyvale`` can be installed into an existing environment of your choice. To create a specific virtual environment for ``pyvale`` navigate to the directory you want to install the environment and use:

.. code-block:: bash

   python3.11 -m venv .pyvale-env
   source .pyvale-env/bin/activate

Standard & Developer Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone ``pyvale`` to your local system along with submodules using

.. code-block:: bash

   git clone --recurse-submodules git@github.com:Computer-Aided-Validation-Laboratory/pyvale.git

``cd`` to the root directory of ``pyvale``. Ensure you virtual environment is activated and run the following commmand from the ``pyvale`` directory:

.. code-block:: bash

   pip install .
   pip install ./dependencies/mooseherder

To create an editable/developer installation of ``pyvale`` and ``mooseherder`` - follow the instructions for a standard installation but run:

.. code-block:: bash

   pip install -e .
   pip install -e ./dependencies/mooseherder

MOOSE
~~~~~~~~~~~~~~~~~~~~~~~

``pyvale`` come pre-packaged with example ``moose`` physics simulation outputs (as *.e exodus files) to demonstrate its functionality. If you need to run additional simulation cases we recommend ``proteus`` (https://github.com/aurora-multiphysics/proteus) which has build scripts for common linux distributions.
