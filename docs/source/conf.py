import os
from sphinx_gallery.sorting import FileNameSortKey

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pyvale'
copyright = '2025, The CAV Team'
author = 'The CAV Team at United Kingdom Atomic Energy Authority (UKAEA)'
release = '2026.6.0'
version = '2026.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinx.ext.viewcode',
    'sphinx_codeautolink',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.mathjax',
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
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'sg_execution_times.rst',
    # Sphinx-gallery generates untitled index and execution-time pages that
    # are only meant to be linked from the hand-written gallery root page.
    'examples/*/index.rst',
    'examples/*/sg_execution_times.rst',
]

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

# Executing the gallery examples requires every optional renderer and external
# simulator to be installed and can take a long time. Execution is therefore
# opt-in: set PYVALE_DOCS_RUN_EXAMPLES=1 to build galleries with captured
# output. By default gallery pages show code and narrative only.
run_examples = os.environ.get("PYVALE_DOCS_RUN_EXAMPLES", "").lower() in (
    "1", "true", "yes",
)

# These examples should be documented but must not execute during gallery
# builds because they require interactive input or unbundled local data. The
# negative lookahead belongs in filename_pattern rather than ignore_pattern:
# Sphinx-Gallery still generates their pages while excluding their execution.
unsafe_example_names = (
    "ex01_region_of_interest",
    "ex05_dic_challenge",
    "ex06_hrdic",
    "ex08_calibration",
    "ex09_stereo",
    "ex10_stereo_platehole",
    "ex11_dic_chal",
)
unsafe_example_pattern = "|".join(unsafe_example_names)
safe_example_pattern = rf"/(?!({unsafe_example_pattern})\.py$)ex"

sphinx_gallery_conf = {
    # Path to your example scripts
    'examples_dirs': [
        '../../src/pyvale/examples/basicsensorsim',
        '../../src/pyvale/examples/dic',
        '../../src/pyvale/examples/render3d',
        '../../src/pyvale/examples/extsensorsim',
        '../../src/pyvale/examples/mooseherder',
        '../../src/pyvale/examples/render2d',
    ],
    # Path to where to save gallery generated output. Render3D galleries are
    # listed before the 2D image-warp galleries.
    'gallery_dirs': [
        'examples/basicsensorsim',
        'examples/dic',
        'examples/render3d',
        'examples/extsensorsim',
        'examples/mooseherder',
        'examples/render2d',
    ],
    # Generate every example page, but only execute unattended examples.
    'filename_pattern': safe_example_pattern,
    # Private helper modules support examples but are not gallery tutorials.
    'ignore_pattern': r'/(?:_blender_example_tools|_riley_demo_tools)\.py$',
    # Specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # Directory where function granular galleries are stored
    'backreferences_dir': 'examples/gen_modules/backreferences',
    # Modules for which function level galleries are created
    'doc_module': ('pyvale',),
    # Additional options
    'download_all_examples': False,
    'plot_gallery': 'True' if run_examples else 'False',
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
