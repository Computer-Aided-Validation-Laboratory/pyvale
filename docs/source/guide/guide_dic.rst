
.. _guide_dic:

Digital Image Correlation (DIC) Guide
======================================

Digital Image Correlation (DIC) is a technique for measuring
deformation, displacement, and strain by analyzing a sequence of images captured
during a loading process. At its core, DIC tracks the movement of patterns on a
surface—tiny speckles or textures—between images, and from this motion, deduces
how the material has deformed.

Local Subset DIC
----------------
The simplest form of DIC divides the image into small regions called *subsets*.
By comparing the intensity patterns of these subsets across images, we can estimate
their displacement. This approach is computationally efficient and works well
for small deformations, but it can struggle when subsets distort significantly.

Shape Functions
---------------
To model how a subset deforms, we introduce *shape functions*. These are
mathematical mappings that describe how points inside a subset move relative to
each other.  Pyvale supports three commonly used shape functions in DIC. The simplest shape function assumes *rigid* translation (two
parameters: horizontal and vertical shift):

.. math::
    \mathbf{x'}=\left[\begin{array}{l}p_0 \\ p_1\end{array}\right]+\left[\begin{array}{cc}1+p_2 & p_3 \\ p_4 & 1+p_5\end{array}\right] \mathbf{x}

Beyond this there's affine shape functions (6 parameters) that can account for translation, scaling, and shearing:

.. math::
    \mathbf{x'}=\left[\begin{array}{l}p_0 \\ p_1\end{array}\right]+\left[\begin{array}{cc}1+p_2 & p_3 \\ p_4 & 1+p_5\end{array}\right] \mathbf{x}

Finally, Pyvale supports quadratic shape functions (12 parameters)

.. math::

    \mathbf{x'} = \begin{bmatrix} p_0 + (1+p_2)x + p_3 y + p_6 x^2 + p_7 xy + p_8 y^2 \\ p_1 + p_4 x + (1+p_5)y + p_9 x^2 + p_{10} xy + p_{11} y^2 \end{bmatrix}

.. image:: shape_functions_light.png
   :class: only-light
   :alt: Diagram (light)


.. image:: shape_functions_dark.png
   :class: only-dark
   :alt: Diagram (dark)

Cost Functions / Correlation Criterion
---------------------------------------
How do we decide if two subsets *match*? To do this we use what's known as a cost function, or often reffered to as a Correlation Criterion.
This is a numerical measure of similarity between the reference subset and the deformed
subset. Pyvale supports three choices

- **Zero-Normalized Sum of Squared Differences (ZNSSD):** Handles both brightness and contrast shifts.
- **Normalized Sum of Squared Differences (NSSD):** More robust to intensity scaling.
- **Sum of Squared Differences (SSD):** Simple, but sensitive to lighting changes.

The goal is to minimize this cost function by adjusting the displacement and
shape parameters until the subsets align as closely as possible.

Cost Function Optimization
----------------------------
Minimizing the value fo the cost function is requires using a optimization routine. 
These algorithms start with an initial guess and refine it step by step. Convergence depends on a good
initial guess and a generally well-behaved cost surface. Pyvale uses a
Levenberg-Marquardt non-linear optimization routine to minimize the cost
function.


Sub-pixel Accuracy
------------------


Sub-pixel Accuracy
------------------
Digital Image Correlation aims for precision beyond the pixel grid. Integer-pixel
matching is a good start, but real displacements rarely align perfectly with pixel
boundaries. To capture these subtle shifts, we refine the measurement to *sub-pixel*
accuracy.

This refinement relies on interpolation. Instead of treating the image as a discrete
array, we approximate it as a smooth surface. Bicubic B-spline interpolation is a
common choice because it provides continuity in both first and second derivatives,
which improves numerical stability during optimization.

### Bicubic B-spline Interpolation
Let the reference image intensity be \( f(x, y) \), known at integer coordinates.
To evaluate it at a non-integer point \((x', y')\), we compute:

.. math::
    f(x', y') \approx \sum_{i=-1}^{2} \sum_{j=-1}^{2} B(x' - (x+i)) \, B(y' - (y+j)) \, f(x+i, y+j)

where:
- \( w_i(x') \) and \( w_j(y') \) are cubic B-spline basis functions,
- the summation spans a \(4 \times 4\) neighborhood around \((x', y')\).

The cubic B-spline basis ensures smooth interpolation:

.. math::
    B(t) = \frac{1}{6}
    \begin{cases}
    (3 - t)^3, & 0 \le t < 1 \\
    (3 - t)^3 - 4(2 - t)^3, & 1 \le t < 2 \\
    (3 - t)^3 - 4(2 - t)^3 + 6(1 - t)^3, & 2 \le t < 3 \\
    0, & t \ge 3
    \end{cases}

This smoothness is essential for gradient-based optimization methods used in DIC.
Linear interpolation, by contrast, introduces discontinuities that can hinder
convergence.

Reliability Guided DIC (RG-DIC)
--------------------------------
Not all subsets are created equal. Some have rich texture and produce reliable
matches; others are bland and ambiguous. RG-DIC tackles this by prioritizing
high-confidence regions first, then propagating their solutions to neighbors.
Think of it as solving the easy puzzles first, then using those clues to crack
the harder ones. This strategy improves robustness, especially in noisy or
low-texture areas.

Strain Calculations
--------------------

Strain Tensor Formulations
-------------------
Once we know how subsets move, we can compute strain. But strain is subtle: it’s
not just displacement, but how displacement gradients vary across space. Common
formulations include:

- **Lagrangian strain:** Based on the original configuration—good for large
  deformations.
- **Eulerian strain:** Based on the current configuration—useful for incremental
  analysis.
- **Green-Lagrange strain:** A nonlinear measure that handles big rotations and
  stretches gracefully.

Choosing the right strain measure depends on your application: small elastic
strains? Stick with linear. Large plastic deformations? Go nonlinear.


