.. _install_windows:

Windows 11
==========

Installing python
-----------------

To be compatible with ``bpy`` (the Blender python interface), ``pyvale`` uses python 3.11. To install python 3.11 for windows grab the correct installer for your system (most likely windows 64 bit) from the python website here: https://www.python.org/downloads/release/python-3119/.

Launch the installer, select 'custom installation' and make sure the 'py launcher' option is checked. Then click next and finish the installation. If you are prompted to disable the maximum path length then click this to confirm you want to disable the path length.

Starting Python with the py Launcher
-------------------------------------

To confirm your installation has worked you should open a windows command line using 'windows-key+r' and then type 'cmd'. From here enter the following to start an interactive python 3.11 interpreter:

.. code-block:: batch

   py -3.11

If you see prompt with Python 3.11.9 in the header and the starting cursor has '>>>' then everything has worked. If you install other python versions on windows you can start them using the py launcher using this syntax ``py -#.#``. Where the '\#' is the version you want to start. For pyvale we will stick with python 3.11.

Troubleshooting the py launcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have trouble starting the py launcher it is most likely because the py launcher is not on the user path. You can fix this with the following steps:

- Open the start menu by clicking on the windows icon, next to the power button click the settings icon which is a gear.
- In the search bar for settings in the top right corner type 'View Advanced System Settings', click on this option to open a window with tabs.
- Go to the 'Advanced' tab and at the bottom click the 'Environment Variables' button.
- In the 'User Variables' box at the top go to the 'PATH' and click edit.
- Add the path to the py launcher by clicking 'new'. The path will likely be: C:\\Users\\YOURUSERNAME\\AppData\\Local\\Programs\\Python\\Launcher\\, replacing YOURUSERNAME with your windows username.

You should now restart your system and try opening a python interpreter through the windows command line again.

Virtual Environment
------------------------

We recommend installing ``pyvale`` in a virtual environment using ``venv`` or ``pyvale`` can be installed into an existing environment of your choice. To create a specific virtual environment for ``pyvale`` first open a windows command prompt ('windows-key+r', type 'cmd', enter). Then navigate to the folder you want to install the environment in (using dir) and use:

.. code-block:: batch

   py -3.11 -m venv pyvale-env

This will create a virtual environment called 'pyvale-env' in a folder of the same name. To activate the virtual environment from your current location type this into the command prompt:

.. code-block:: batch

   pyvale-env\Scripts\activate

If this has worked you should see '(pyvale-env)' at the start of your command prompt line showing the environmen is activated. If you ever need to activate the environment in a new command prompt you just need to run the 'activate' script again from the folder you are currently in.


Installation from PyPI
------------------------
``pyvale`` can be installed from PyPI. Ensure you virtual environment is activated (you should see '(pyvale-env)' in your command prompt) and run the following:

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

To use you virtual environment in VSCode to run some of the examples you will need to make sure you have selected your virtual environment as your python interpreter. To do this first open the folder that you want to work from which should be the same folde that contains your virtual environment folder (that is the pyvale-env folder). Now go to the search bar at the top or open the command palette using ctrl+shift+p and type 'Python: Select Interpreter'. You should see your virtual environment listed which you can select. Now when you run python scripts VSCode should automatically use your virtual envvironment.

Installation from Source
------------------------
This will only be needed if you want an editable installation of ``pyvale`` for most applications users will want to use the PyPI version above.

Clone ``pyvale`` to your local system using git along with submodules using:

.. code-block:: bash

   git clone git@github.com:Computer-Aided-Validation-Laboratory/pyvale.git

For this case it is normally easier to keep your virtual environment stored in the ``pyvale`` folder so create a virtual environment there first. Then, ensure you virtual environment is activated and run the following commmand from the ``pyvale`` folder:

.. code-block:: bash

   pip install -e .

This will create an editable/developer installation of ``pyvale``.
