.. _install_linux:

Ubuntu Linux
=============

Managing Python Versions
------------------------

To be compatible with ``bpy`` (the Blender python interface), ``pyvale`` uses python 3.11. To install python 3.11 without corrupting your operating systems python installation first add the deadsnakes repository to apt:

.. code-block:: bash

   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update && sudo apt upgrade -y

Install python 3.11:

.. code-block:: bash

   sudo apt install python3.11
   sudo apt install python3.11-dev

Add ``venv`` to your python 3.11 install:

.. code-block:: bash

   sudo apt install python3.11-venv

Check your python 3.11 install is working using the following command which should open an interactive python interpreter:

.. code-block:: bash

   python3.11

If everything has worked you should see an interactive python console with Python 3.11.xx in the header. You can now exit the interpreter using ``quit()``.

Virtual Environment
------------------------

We recommend installing ``pyvale`` in a virtual environment using ``venv`` or ``pyvale`` can be installed into an existing environment of your choice. To create a specific virtual environment for ``pyvale`` navigate to the directory you want to install the environment and use:

.. code-block:: bash

   python3.11 -m venv pyvale-env

Now activate the virtual environment:

.. code-block:: bash

   source pyvale-env/bin/activate

If you need to activate the environment again in a new terminal use the above command from the directory containing the 'pyvale-env' directory.

Installation from PyPI
------------------------
``pyvale`` can be installed from PyPI. Ensure you virtual environment is activated (you should see '(pyvale-env)' in your terminal) and run the following from the ``pyvale`` directory:

.. code-block:: bash

   pip install pyvale

You should now be able to start a python 3.11 interpreter in your terminal using (again make sure your pyvale-env is active):

.. code-block:: bash

   python

Now check that you can import pyvale in the interpreter:

.. code-block:: python

   import pyvale

If there are no errors then everything has worked and you can now move on to looking at some of our examples to get you started in the basics section.


Installation from Source
------------------------
This will only be needed if you want an editable installation of ``pyvale`` for most applications users will want to use the PyPI version above.

Clone ``pyvale`` to your local system along with submodules using

.. code-block:: bash

   git clone --recurse-submodules git@github.com:Computer-Aided-Validation-Laboratory/pyvale.git

``cd`` to the root directory of ``pyvale``. Ensure you virtual environment is activated and run the following commmand from the ``pyvale`` directory:

.. code-block:: bash

   pip install -e .
   pip install -e ./dependencies/mooseherder

This will create an editable/developer installation of ``pyvale`` and ``mooseherder``.

.. MOOSE
.. ------------------------

.. ``pyvale`` come pre-packaged with example ``moose`` physics simulation outputs (as *.e exodus files) to demonstrate its functionality. If you need to run additional simulation cases we recommend ``proteus`` (https://github.com/aurora-multiphysics/proteus) which has build scripts for common linux distributions.
