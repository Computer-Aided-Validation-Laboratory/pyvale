.. _install_mac:

MacOS
=============

Installing ``python3.11`` using the homebrew package manager:
--------------------------------------------------------------

To be compatible with ``bpy`` (the Blender python interface), ``pyvale`` requires python 3.11. 
Homebrew is a free and open-source commonly used package manager for macOS that often simplifies the process of installing, updating, and managing software. 
Simple installation instructions for Homebrew be found at `<https://brew.sh/>`_.

After you have homebrew setup. To install python3.11 open a terminal and enter
the command:

.. code-block::
   
    brew install python@3.11

To verify the installation you can try the command:

.. code-block::

   which python3.11

Installing ``python3.11`` from python.org
-------------------------------------------

To be compatible with ``bpy`` (the Blender python interface), ``pyvale`` requires python 3.11. 
To install python3.11 from `<https://python.org>`_ hover over the “Downloads” link in the navigation. Select macOS.
You'll then need to select 3.11.9 (or whatever the latest version of 3.11 is).
This should download the installer which you can then run through.

If you have multiple python version the above instructions should put python in
location ``/Library/Frameworks/Python.framework/Versions/3.11/bin/``. In that
folder there should be a ``python3``, which is really a link to ``python3.11``.


Virtual Environment
------------------------

We recommend installing ``pyvale`` in a virtual environment using ``venv`` or ``pyvale`` can be installed into an existing environment of your choice. 
To create a specific virtual environment for ``pyvale`` first open a terminal Then navigate to the folder you want to install the environment in (using ``cd``) and use:

.. code-block:: batch

   python3.11 -m venv pyvale-env

This will create a virtual environment called 'pyvale-env' in a folder of the same name. To activate the virtual environment from your current location type this into your terminal:

.. code-block:: batch

   source pyvale-env/bin/activate

If this has worked you should see '(pyvale-env)' at the start of your command prompt line showing the environmen is activated. If you ever need to activate the environment in a new command prompt you just need to run the 'activate' script again from the folder you are currently in.


Installation from PyPI
------------------------
``pyvale`` can be installed from PyPI. Ensure you virtual environment is activated (you should see '(pyvale-env)' terminal) and run the following:

.. code-block:: batch

   pip install pyvale

You should now be able to start a python 3.11 interpreter in your terminal using (again make sure your pyvale-env is active):

.. code-block:: bash

   python

Now check that you can import pyvale in the interpreter:

.. code-block:: python

   import pyvale

If there are no errors then everything has worked and you can now move on to looking at some of our examples to get you started in the basics section.

Virtual Environments in VSCode
------------------------------

To use you virtual environment in VSCode to run some of the examples you will need to make sure you have selected your virtual environment as your python interpreter. 
To do this first open the folder that you want to work from which should be the same folde that contains your virtual environment folder (that is the pyvale-env folder). 
Now go to the search bar at the top or open the command palette using cmd+shift+P and type 'Python: Select Interpreter'. 
You should see your virtual environment listed which you can select. Now when you run python scripts VSCode should automatically use your virtual envvironment.

Installation from Source
------------------------
This will only be needed if you want an editable installation of ``pyvale`` for most applications users will want to use the PyPI version above. 
In addition to setting up a virtual environment as shown above, it's reccomended you install ``clang`` and ``libomp`` using the homebrew (`https://brew.sh/`_) package manager:

.. code-block:: bash

   brew install clang
   brew install libomp

You'll then need to ensure that we use the compilers installed from homebrew:

.. code-block:: bash

   export CC=/opt/homebrew/opt/llvm/bin/clang
   export CXX=/opt/homebrew/opt/llvm/bin/clang++

Clone ``pyvale`` to your local system using git along with submodules using:

.. code-block:: bash

   git clone git@github.com:Computer-Aided-Validation-Laboratory/pyvale.git

For this case it is normally easier to keep your virtual environment stored in the ``pyvale`` folder so create a virtual environment there first. Then, ensure you virtual environment is activated and run the following commmand from the ``pyvale`` folder:

.. code-block:: bash

   pip install -e .

This will create an editable/developer installation of ``pyvale``.


