.. pyvale documentation master file, created by
   sphinx-quickstart on Wed Apr  2 13:15:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyvale documentation
====================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

pyvale
----------------

.. automodule:: pyvale

.. autofunction:: pyvale.create_sensor_pos_array

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. py:function:: lumache.get_random_ingredients(kind=None)

   Return a list of random ingredients as strings.

   :param kind: Optional "kind" of ingredients.
   :type kind: list[str] or None
   :return: The ingredients list.
   :rtype: list[str]



