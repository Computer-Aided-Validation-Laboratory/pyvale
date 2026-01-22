

.. _guide_dic:

Digital Image Correlation (DIC) Theory Overview
======================================

Digital Image Correlation (DIC) is a technique for measuring
deformation, displacement, and strain by analyzing a sequence of images captured
during a loading process. At its core, DIC tracks the movement of patterns/speckles/textures between images, 
and from this motion, deduces how the material has deformed.

Local Subset DIC
----------------
In Local subset DIC images are divided into small :math:`N \times N` pixel regions which are called *subsets*.
By comparing the intensity patterns of these subsets across images, we can estimate
their displacement. The image below shows the difference/cost (where 0 is a perfect match)
for a brute force scan along the x-axis. 


.. image:: cost.gif
    :alt: Example Cost Minimization

Typically we don't use a brute forcea approach, but instead use an **optimization algorith** that is much more computationally
efficient. The optimization tries to minimize the difference between the
reference and deformed subset by using the gradient. You can see from the
brute-force approach that there's local minima where the optimizer might get
stuck. It's therefore important to ensure that any initial guess is reasonably
close to the actual parameters that define the mapping from the subset in the
reference image to the subset in the deformed image.

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

where :math:`f(x_{i},y_{i})` and  :math:`g(x_{i},y_{i})` represent the gray-level intensity values at location :math:`(x_{i},y_{i})` in the reference and deformed images, respectively. 
To reduce sensitivity to intensity scaling, the Normalized SSD (NSSD) is defined as:

.. math::
  \text{NSSD} = \sum_i \left( \frac{f(x_{i},y_{i})}{\sqrt{\sum_{j} f(x_{j},y_{j})^{2}}} - \frac{g(x_{i},y_{i})}{\sqrt{\sum_{j} g(x_{j},y_{j})^{2}}} \right)^{2}

Finally there's the ZNSSD, which is invariant to both additive and multiplicative intensity changes:

.. math::
  \text{ZNSSD} = \sum_i \left( \frac{\bar{f}(x_{i},y_{i})}{\sqrt{\sum_{j} \bar{f}(x_{j},y_{j})^{2}}} - \frac{\bar{g}(x_{i},y_{i})}{\sqrt{\sum_{j} \bar{g}(x_{j},y_{j})^{2}}} \right)^{2}

where :math:`\bar{f}(x_{i},y_{i}) = f(x_{i},y_{i}) - f_m`, and :math:`f_m` is the mean gray-level value of the subset. 

Cost Function Optimization
----------------------------
Minimizing the value fo the cost function is requires using a optimization routine. 
These algorithms start with an initial guess and refine it step by step. Convergence depends on a good
initial guess and a generally well-behaved cost surface. 
Pyvale uses a **Levenberg-Marquardt** non-linear optimization routine to minimize the cost function.
The algorithm proceeds iteratively by updating the shape-function parameter
vector :math:`\mathbf{p}`. At iteration :math:`i`, the parameters are updated according to

.. math::

   \mathbf{p}_{i+1}
   = \mathbf{p}_{i}
     - \left( \mathbf{H} + \lambda\,\mathrm{diag}\!\left[\mathbf{H}\right] \right)^{-1}
       \nabla C(\mathbf{p}_{i}),

where :math:`\nabla C(\mathbf{p}_{i})` denotes the gradient of the cost function
evaluated at the current parameters. The matrix :math:`\mathbf{H}` is an
approximation to the Hessian, taken here as
:math:`\mathbf{H} \approx \mathbf{J}\mathbf{J}^{\mathsf{T}}`, with
:math:`\mathbf{J}` the Jacobian of the residuals with respect to
:math:`\mathbf{p}`. The scalar :math:`\lambda > 0` is the
Levenberg–Marquardt damping factor, which controls the balance between
gradient-descent and Gauss–Newton behaviour.

After computing the updated parameter vector :math:`\mathbf{p}_{i+1}`, the cost
function is re-evaluated to assess the quality of the update. Based on the
resulting change in error, the damping parameter :math:`\lambda` is adjusted. 
If the cost **increases**, don't update the shape function parameters and
**increase** the damping by a factor of 10. 
If the cost **decreases**, the update is accepted and the damping is **reduced** by a factor of 10.
This update–evaluation–adjustment cycle is repeated until convergence criteria
based on both parameter change and cost-function reduction are satisfied. The
magnitude of the parameter update is quantified by

.. math::

   \text{d}p_{\text{norm}}
     = \sqrt{\langle \Delta\mathbf{p}, \Delta\mathbf{p} \rangle},
   \qquad
   p_{\text{norm}}
     = \sqrt{\langle \mathbf{p}, \mathbf{p} \rangle},

where :math:`\Delta\mathbf{p} = \mathbf{p}_{i+1} - \mathbf{p}_{i}`, and
:math:`\langle \cdot,\cdot \rangle` denotes the Euclidean inner product. From
these quantities, the relative parameter-update tolerance is defined as

.. math::

   x_{\text{tol}}
     = \frac{\text{d}p_{\text{norm}}}{p_{\text{norm}} + \varepsilon},

with :math:`\varepsilon` a small regularizing constant included to avoid division
by zero. Convergence in terms of the cost function is measured using

.. math::

   f_{\text{tol}}
     = \frac{\lvert C_{i+1} - C_{i} \rvert}{\lvert C_{i} \rvert + \varepsilon}.

Iteration terminates once both :math:`x_{\text{tol}}` and
:math:`f_{\text{tol}}` fall below user-specified thresholds, indicating that
further updates would produce only negligible changes in the parameters and the
cost function.


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

.. list-table::
   :width: 70%
   :class: borderless

   * - .. image:: ./guide_preinterp.png
          :width: 100%
         
     - .. image:: ./guide_interp.png
          :width: 100%

Reliability Guided DIC (RG-DIC)
--------------------------------
It's highly likely that some subsets will poorly correlate due to texture changes, cracks, noise, changes in lighting, or large local deformations.
Reliability‑Guided DIC (RG‑DIC) addresses this by correlating subsets in an order determined by the correlation coefficient.
After selecting and correlating at aseed location, RG‑DIC uses the results to provide initial guesses for the correlation of neighbouring subsets. 
The algorithm then expands outward in a front‑propagation style. 
Mathematical details can be found `here <https://opg.optica.org/ao/abstract.cfm?uri=ao-48-8-1535>`_.

.. image:: guide_rgdic.gif
   :width: 80%
   :align: center

Strain Calculations & Formulations
-----------------------------------
The in-plane strain components can be calculated using the spatial derivatives of the displacement field to capture local changes in geometry. 
The 2D deformation gradient matrix, :math:`\mathbf{F}`, is given by:

.. math::

   \mathbf{F} =
   \begin{bmatrix}
     1+\dfrac{\partial u_x}{\partial x} & \dfrac{\partial u_x}{\partial y} \\
     \dfrac{\partial u_y}{\partial x} & 1+\dfrac{\partial u_y}{\partial y}
   \end{bmatrix}
   = \mathbf{I} + \nabla \boldsymbol{u},
   \qquad
   \boldsymbol{u} =
   \begin{bmatrix}
     u_x \\ u_y
   \end{bmatrix}.

To calculate the partial derivatives above, we compute the gradient over a square window containing :math:`N \times N` displacement data points from the DIC calculation. 
Because we are using displacement values from DIC, the strain window is dependent on the subset-step, :math:`s`, and subset-size, :math:`w`. 
These quantities, along with the size of the strain window, form what is typically referred to as the Virtual Strain Gauge, :math:`VSG`:

.. math::

   VSG = (\mathrm{N} - 1)s + w

Due to the noise in DIC measurements, smoothing is typically applied over the strain window. 
We supports bilinear and biquadratic smoothing over the strain window elements. 
The polynomial approximation is given by:

.. math::

   \boldsymbol{u}(x,y) = \mathbf{P}(x,y)\,\mathbf{c},

where :math:`\boldsymbol{u} = [\,u_x\;u_y\,]^{T}`, :math:`\mathbf{P}(x,y)` is a row vector of basis terms, and :math:`\mathbf{c}` is a column vector of coefficients. The polynomial basis is:

.. math::

   \mathbf{P}(x,y) =
   \begin{cases}
     [1, x, y] & \text{bilinear}, \\
     [1, x, y, x^2, y^2, x^2 y, x y^2, x^2 y^2] & \text{biquadratic}
   \end{cases}

The coefficients are obtained by solving a linear least-squares problem, since the displacement field is linearly parameterized in the unknown coefficients. 
Once the polynomial coefficients have been obtained, the deformation gradient tensor and strain can be calculated. We currently support the following strain tensor formulations:

- ``strain_formulation="HENCKY"``, *Hencky (logarithmic)*: :math:`\boldsymbol{\varepsilon}=\ln(\sqrt{\mathbf{F}^\mathsf{T}\mathbf{F}})`
- ``strain_formulation="GREEN"``, *Green–Lagrange*: :math:`\boldsymbol{\varepsilon}=\tfrac{1}{2}\left(\mathbf{F}^\mathsf{T}\mathbf{F}-\mathbf{I}\right)`
- ``strain_formulation="ALMANSI"``, *Euler–Almansi*: :math:`\boldsymbol{\varepsilon}=\tfrac{1}{2}\left(\mathbf{I}-(\mathbf{F}\mathbf{F}^\mathsf{T})^{-1}\right)`
- ``strain_formulation="BIOT_LAGRANGE"``, *Biot (right / Lagrangian)*: :math:`\boldsymbol{\varepsilon}=\sqrt{\mathbf{F}^\mathsf{T}\mathbf{F}}-\mathbf{I}`
- ``strain_formulation="BOIT_EULER"``, *Biot (left / Eulerian)*: :math:`\boldsymbol{\varepsilon}=\sqrt{\mathbf{F}\mathbf{F}^\mathsf{T}}-\mathbf{I}`

where :math:`\mathbf{I}` is the :math:`2 \times 2` identity matrix.
