import os
import sys
import inspect
from sphinx_gallery.sorting import FileNameSortKey

# Add source paths
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
release = '2026.1.3'
version = '2026.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',  # Adds source code links
    'sphinx_codeautolink',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
    'breathe',
    'myst_parser'
]

# Language settings
language = 'en'

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Master document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Napoleon configuration (NumPy style docstrings) ------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False  # Disable Google style since you use NumPy
napoleon_use_rtype = False
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_attr_annotations = True

# -- Autodoc configuration --------------------------------------------------

autodoc_typehints = 'none'  # Don't show type hints in signatures
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__',
    'inherited-members': False,
    'show-inheritance': True
}
autodoc_inherit_docstrings = True

# Prevent duplication issues
autoclass_content = 'class'  # Only class docstring, not __init__
add_module_names = False  # Keep class names short




# -- Autosummary configuration ----------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False




# -- Breathe configuration (for C++ docs) -----------------------------------
breathe_projects = {"pyvale": "./doxygen/xml"}
breathe_default_project = "pyvale"




# -- Code autolink configuration --------------------------------------------
codeautolink_concat_default = True





# -- Sphinx Gallery configuration -------------------------------------------
sphinx_gallery_conf = {
    # Path to your example scripts
    'examples_dirs': [
        '../../src/pyvale/examples/basicsensorsim',
        '../../src/pyvale/examples/extsensorsim',
        '../../src/pyvale/examples/dic',
        '../../src/pyvale/examples/blenderimagedef',
        '../../src/pyvale/examples/mooseherder',
    ],
    # Path to where to save gallery generated output
    'gallery_dirs': [
        'examples/basicsensorsim',
        'examples/extsensorsim',
        'examples/dic',
        'examples/blenderimagedef',
        'examples/mooseherder',
    ],
    # Pattern to identify example files
    'filename_pattern': '/plot_',
    # Specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # Directory where function granular galleries are stored
    'backreferences_dir': 'examples/gen_modules/backreferences',
    # Modules for which function level galleries are created
    'doc_module': ('pyvale',),
    # Additional options
    'download_all_examples': False,
    'plot_gallery': 'True',
    'remove_config_comments': True,
    'expected_failing_examples': [],
    'show_memory': False,
    'show_signature': True,
}




# -- Copy button configuration ----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True




# -- MyST Parser configuration ----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for HTML output ------------------------------------------------

html_theme = 'furo'
html_title = "Pyvale: The Python Validation Engine"
html_favicon = "_static/pyvale_logo_badge.png"

# Theme options
html_theme_options = {
    "light_logo": "pyvale_logo_badge.png",
    "dark_logo": "pyvale_logo_badge.png",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/Computer-Aided-Validation-Laboratory/pyvale/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# Static files
html_static_path = ['_static']
html_css_files = ["custom.css"]

# Code highlighting
pygments_style = 'default'
pygments_dark_style = 'monokai'

# -- Options for LaTeX output -----------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
''',
}

latex_documents = [
    (master_doc, 'pyvale.tex', 'Pyvale Documentation',
     'The CAV Team', 'manual'),
]

# -- Options for manual page output -----------------------------------------

man_pages = [
    (master_doc, 'pyvale', 'Pyvale Documentation',
     [author], 1)
]

# -- Options for Texinfo output ---------------------------------------------

texinfo_documents = [
    (master_doc, 'pyvale', 'Pyvale Documentation',
     author, 'pyvale', 'The Python Validation Engine',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Additional settings for better documentation
nitpicky = False
nitpick_ignore = []

# Suppress warnings
suppress_warnings = ['image.nonlocal_uri']

# -- Custom settings for better NumPy docstring handling -------------------

# Ensure Napoleon processes docstrings before other extensions
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'array_like': 'array-like',
    'array-like': 'array-like',
    'ndarray': '~numpy.ndarray',
}

# -- Custom CSS and JS files ------------------------------------------------
html_js_files = []

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
