

.. _guide_dic:

Digital Image Correlation (DIC) Guide
======================================

Digital Image Correlation (DIC) is a technique for measuring
deformation, displacement, and strain by analyzing a sequence of images captured
during a loading process. At its core, DIC tracks the movement of patterns/speckles/textures between images, 
and from this motion, deduces how the material has deformed.

Local Subset DIC
----------------
In Local subset DIC images are divided into small :math:`N \times N` pixel regions which are called *subsets*.
By comparing the intensity patterns of these subsets across images, we can estimate
their displacement. The image below shows the cost (where 0 is a perfect match)
for a brute force scan along the x-axis. 

Typically we don't use a brute force
approach, but instead use an optimization algorith that is much more computationally
efficient. The optimization tries to minimize the difference between the
reference and deformed subset by using the gradient. You can see from the
brute-force approach that there's local minima where the optimizer might get
stuck. It's therefore important to ensure that any initial guess is reasonably
close to the actual parameters that define the mapping from the subset in the
reference image to the subset in the deformed image.


.. image:: cost.gif
    :alt: Example Cost Minimization


Shape Functions
---------------
It's often the case that a subset undergoes a more complex translation than just a pure rigid translation between reference and deformed image.
To model how a subset deforms more genreally, we introduce *shape functions*. These are
mathematical mappings that describe how points inside a subset move relative to
each other.  Pyvale supports three commonly used shape functions in DIC. The simplest of course is a pure *rigid* translation (two
parameters: horizontal and vertical shift). After this comes *affine* and
*quadratic* shape functions:

.. math::
  \xi(x_i,y_i, \mathbf{p}) =
  \underbrace{\begin{bmatrix} p_0 \\ p_1 \end{bmatrix}}_{\text{rigid}}
  + \underbrace{\begin{bmatrix} 1+p_2 & p_3 \\ p_4 & 1+p_5 \end{bmatrix} 
  \begin{bmatrix} x_i \\ y_i \end{bmatrix}}_{\text{affine}} + \underbrace{\begin{bmatrix} p_6 & p_7 & p_8 \\ p_9 & p_{10} & p_{11} \end{bmatrix} 
  \begin{bmatrix} x_i^2 \\ x_iy_i \\ y_i^2 \end{bmatrix}}_{\text{quadratic}}

Each higher-order shape function includes all terms from the lower-order functions: affine includes rigid terms, and quadratic includes both affine and rigid terms. 

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

.. math::
  \text{SSD} = \sum_i \big(f(x_{i},y_{i}) - g(x_{i},y_{i})\big)^{2}

where :math:`f(x_{i},y_{i})` and  :math:`g(x_{i},y_{i})` represent the gray-level intensity values at location :math:`(x_{i},y_{i})` in the reference and deformed images, respectively. To reduce sensitivity to intensity scaling, the Normalized SSD (NSSD) is defined as:

.. math::
  \text{NSSD} = \sum_i \left( \frac{f(x_{i},y_{i})}{\sqrt{\sum_{j} f(x_{j},y_{j})^{2}}} - \frac{g(x_{i},y_{i})}{\sqrt{\sum_{j} g(x_{j},y_{j})^{2}}} \right)^{2}

.. math::
  \text{ZNSSD} = \sum_i \left( \frac{\bar{f}(x_{i},y_{i})}{\sqrt{\sum_{j} \bar{f}(x_{j},y_{j})^{2}}} - \frac{\bar{g}(x_{i},y_{i})}{\sqrt{\sum_{j} \bar{g}(x_{j},y_{j})^{2}}} \right)^{2}

where :math:`\bar{f}(x_{i},y_{i}) = f(x_{i},y_{i}) - f_m`, and :math:`f_m` is the mean gray-level value of the subset. ZNSSD is invariant to both additive and multiplicative intensity changes.
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
DIC aims for precision beyond the integer values defined by the images pixel grid. Integer-pixel
matching is a good start, but physical displacements do not perfectly map to pixel
boundaries. To capture this, we refine the measurement to *sub-pixel*
accuracy using interpolation. Instead of treating the image as a discrete
array, we approximate it as a smooth surface. This is done using **cubic B-spline
interpolation**. Generally, B-spline curves do not pass through the control
points (pixel values) and *interpolating* B-splines are needed to pass through
exact locations. More details can be found `here <https://ieeexplore.ieee.org/document/1163154>`_.

Reliability Guided DIC (RG-DIC)
--------------------------------
It's highly likely that some subsets will poorly correlate due to texture changes, occlusion, noise, or large local deformations. 
Reliability‑Guided DIC (RG‑DIC) addresses this by correlating subsets in an order determined by the correlation coefficient.
After selecting and correlating at aseed location, RG‑DIC uses the results to provide initial guesses for the correlation of neighbouring subsets. 
The algorithm then expands outward in a front‑propagation style. 
Mathematical details can be found `here <https://opg.optica.org/ao/abstract.cfm?uri=ao-48-8-1535>`_.

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


