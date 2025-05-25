import os 
import sys
import inspect
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath('../../src/pyvale/'))
sys.path.insert(0, os.path.abspath('../../src/pyvale/dic/'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pyvale'
copyright = '2025, The CAV Team'
author = 'The CAV Team at United Kingdom Atomic Energy Authority (UKAEA)'
release = '2025.4.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.linkcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx_codeautolink',
              'sphinx_copybutton',
              'sphinx_gallery.gen_gallery',
              'breathe',
              'myst_parser']

napoleon_numpy_docstring = True
napoleon_use_rtype = False
autodoc_typehints = 'signature'

breathe_projects = {"test": "./doxygen/xml"}
breathe_default_project = "test"


# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../../src/pyvale/examples/basics',
                      '../../src/pyvale/examples/renderblender'],

    # path to where to save gallery generated output
    'gallery_dirs': ['examples/basics',
                     'examples/renderblender'],
    
    'filename_pattern': '/plot_', 
    # specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # directory where function granular galleries are stored
    'backreferences_dir': 'examples/gen_modules/backreferences',
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    'doc_module': ('pyvale'),
}

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ["custom.css"]

html_theme_options = {
    "light_logo": "logo_no_background.png",
    "dark_logo": "logo_no_background_inverted.png",
}
html_title = "The Python Validation Engine"

autosummary_generate = True
autoclass_content = "both"



def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # Get the source file and line number
    try:
        source_file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
        line = lines[1]
    except Exception:
        return None

    filename = os.path.relpath(source_file, start=os.path.abspath('src'))
    return f"https://github.com/Computer-Aided-Validation-Laboratory/pyvale/tree/main/src/pyvale/{filename}#L{line}"

