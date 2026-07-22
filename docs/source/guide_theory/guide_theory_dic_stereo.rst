.. _guide_theory_dic_stereo:

Stereo DIC
======================================

Stereo Digital Image Correlation (Stereo DIC) extends 2D DIC by using two
synchronised cameras instead of one. The idea is to measure the same material
point in the left and right image, then use the calibrated camera geometry to
triangulate its 3D position. Repeating this for each deformed image gives the
3D displacement field.

In Pyvale, Stereo DIC still uses the same subset matching, shape functions,
correlation criteria, interpolation, and Levenberg-Marquardt optimization
covered in the :ref:`DIC theory overview <guide_theory_dic>`. The extra pieces
are the stereo camera geometry, the left-right matching strategy, and the
conversion from pixel coordinates to world coordinates.

Stereo Processing Strategy
--------------------------
The stereo workflow can be thought of as two linked DIC problems. The left
camera is used for the usual temporal correlation between the reference and
current deformed image. Once the left-image displacement is known, Pyvale uses
that result to find the matching subset in the right camera image.

For a subset with reference left-image centre :math:`(x_l^0,y_l^0)`, the left
camera temporal DIC gives a deformed left-image centre

.. math::

   x_l^k = x_l^0 + u_l^k, \qquad
   y_l^k = y_l^0 + v_l^k,

where :math:`u_l^k` and :math:`v_l^k` are the 2D DIC displacement components for
image :math:`k`. The stereo match then searches for the corresponding point in
the right image,

.. math::

   x_r^k = x_l^0 + u_r^k, \qquad
   y_r^k = y_l^0 + v_r^k.

Here :math:`u_r^k` and :math:`v_r^k` are stored as the stereo pixel displacement
components. They describe where the matching right-image subset is relative to
the original left-image subset grid.

.. image:: guide_theory_dic_stereo_light.png
   :class: only-light
   :alt: Diagram (light)

.. image:: guide_theory_dic_stereo_dark.png
   :class: only-dark
   :alt: Diagram (dark)

Camera Geometry
---------------
Stereo DIC needs a calibration for both cameras. Pyvale uses each camera's
intrinsic matrix,

.. math::

   \mathbf{K} =
   \begin{bmatrix}
   f_x & f_s & c_x \\
   0   & f_y & c_y \\
   0   & 0   & 1
   \end{bmatrix},

where :math:`f_x` and :math:`f_y` are the focal lengths in pixels,
:math:`f_s` is the skew term, and :math:`(c_x,c_y)` is the principal point. The
relative camera pose is defined by a rotation matrix :math:`\mathbf{R}` and a
translation vector :math:`\mathbf{t}` from the left camera to the right camera.
In the current implementation the rotation matrix is built from the calibration
Euler angles using the order

.. math::

   \mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x.

From this calibration, Pyvale forms the fundamental matrix

.. math::

   \mathbf{F}
   = \mathbf{K}_1^{-\mathsf{T}}[\mathbf{t}]_\times
     \mathbf{R}\mathbf{K}_0^{-1},

where :math:`[\mathbf{t}]_\times` is the skew-symmetric matrix for the stereo
translation. The fundamental matrix defines the epipolar constraint between the
left and right images.

Epipolar Constraint
-------------------
For a point in the left image written in homogeneous coordinates as
:math:`\mathbf{x}_l = [x_l, y_l, 1]^\mathsf{T}`, the corresponding point in the
right image must lie on the epipolar line

.. math::

   \mathbf{l}_r = \mathbf{F}\mathbf{x}_l.

If :math:`\mathbf{l}_r = [a,b,c]^\mathsf{T}`, then any matching right-image point
:math:`(x_r,y_r)` satisfies

.. math::

   a x_r + b y_r + c = 0.

This is useful because it reduces the initial stereo search from a 2D image
search to a search along a line. Pyvale computes the closest point on the
epipolar line to the left-image subset centre,

.. math::

   \mathbf{P}
   =
   \begin{bmatrix} x_l \\ y_l \end{bmatrix}
   -
   \frac{a x_l + b y_l + c}{a^2+b^2}
   \begin{bmatrix} a \\ b \end{bmatrix},

and the unit direction of the epipolar line,

.. math::

   \mathbf{d}
   =
   \frac{1}{\sqrt{a^2+b^2}}
   \begin{bmatrix} -b \\ a \end{bmatrix}.

This line direction is also used to create a local rectified search coordinate
system. The perpendicular direction is

.. math::

   \mathbf{d}_{\perp} =
   \begin{bmatrix} d_y \\ -d_x \end{bmatrix}.

Initial Guess From Rectified FFT
--------------------------------
Just as 2D DIC benefits from a good initial guess, stereo matching also needs a
starting point that is close to the correct right-image subset. Pyvale does this
by building a small rectified search window around the epipolar line.

The left subset is placed in the centre of an FFT correlation window. The right
image is then sampled in a coordinate system aligned with the epipolar line,

.. math::

   \mathbf{s}(i,j)
   = \mathbf{P} + i\mathbf{d} - j\mathbf{d}_{\perp},

where :math:`i` moves along the epipolar line and :math:`j` moves perpendicular
to it. This gives an unrectified patch from the right image, sampled as if the
local search region had been rectified.

FFT cross-correlation is then used to estimate the translation between the left
subset and this rectified right-image window. If the correlation peak is
:math:`(\Delta i, \Delta j)`, the estimated right-image point is

.. math::

   \mathbf{x}_r
   = \mathbf{P} + \Delta i\mathbf{d} - \Delta j\mathbf{d}_{\perp}.

This gives the initial rigid translation

.. math::

   p_0 = x_r - x_l, \qquad p_1 = y_r - y_l.

For affine or quadratic shape functions, the initial parameters also include the
local coordinate transformation implied by the epipolar-line basis,

.. math::

   \begin{bmatrix}
   1+p_2 & p_3 \\
   p_4   & 1+p_5
   \end{bmatrix}
   =
   \begin{bmatrix}
   d_x & -d_{\perp,x} \\
   d_y & -d_{\perp,y}
   \end{bmatrix}.

This is only an initial estimate. The final match is still obtained using the
same nonlinear subset optimization described in the 2D DIC theory guide.

Reliability-Guided Stereo Matching
----------------------------------
Pyvale uses a reliability-guided strategy for stereo matching in the same spirit
as RG-DIC. Seed subsets are matched first, then the solution expands through the
subset grid using previously matched neighbours as initial guesses.

The algorithm proceeds as follows:

#. The left-image temporal DIC result is used to locate the current subset centre
   in the deformed left image.
#. The current left subset is built using the temporal left-camera shape-function
   parameters.
#. For each seed point, an initial stereo guess is estimated from rectified FFT
   correlation along the epipolar line.
#. The optimizer refines the left-to-right subset match in the right image.
#. Neighbouring subsets are added to a queue ordered by their correlation cost.
#. When a subset is processed, Pyvale uses a successful neighbouring stereo
   result as the initial guess where possible. If the neighbour did not pass the
   threshold, the epipolar FFT estimate is used again.
#. The process expands through the active subset grid until all reachable subsets
   have been attempted.

For deformed images after the first one, the stereo shape-function parameters are
composed with the left-camera temporal parameters. This means the stored stereo
parameters describe the mapping from the original left reference subset through
to the current right-image subset, rather than only the incremental left-to-right
match for that frame.

Triangulation
-------------
Once a left-image point and right-image point have been matched, the 3D position
can be calculated. Pyvale first undistorts both image points using the calibrated
radial and tangential distortion parameters. The distorted pixel point is
converted to normalized camera coordinates, and then an iterative update is used
to remove distortion.

The normalized left and right image points are written as

.. math::

   \mathbf{x}_l = [x_l, y_l, 1]^\mathsf{T}, \qquad
   \mathbf{x}_r = [x_r, y_r, 1]^\mathsf{T}.

Pyvale then uses linear triangulation with projection matrices

.. math::

   \mathbf{P}_0 = [\mathbf{I}\mid\mathbf{0}], \qquad
   \mathbf{P}_1 = [\mathbf{R}\mid\mathbf{t}].

The world point :math:`\mathbf{X} = [X,Y,Z,1]^\mathsf{T}` is found by solving the
homogeneous DLT system

.. math::

   \mathbf{A}\mathbf{X} = \mathbf{0},

where

.. math::

   \mathbf{A} =
   \begin{bmatrix}
   x_l\mathbf{P}_{0,3} - \mathbf{P}_{0,1} \\
   y_l\mathbf{P}_{0,3} - \mathbf{P}_{0,2} \\
   x_r\mathbf{P}_{1,3} - \mathbf{P}_{1,1} \\
   y_r\mathbf{P}_{1,3} - \mathbf{P}_{1,2}
   \end{bmatrix}.

Here :math:`\mathbf{P}_{m,n}` denotes row :math:`n` of projection matrix
:math:`\mathbf{P}_m`. The solution is the right singular vector corresponding to
the smallest singular value of :math:`\mathbf{A}`. After converting from
homogeneous coordinates, Pyvale stores the 3D coordinates
:math:`(X,Y,Z)`. If the stereo translation is supplied in millimetres, these
coordinates are also in millimetres.

3D Displacement
---------------
For the first image pair, the triangulated coordinates are stored as the stereo
reference coordinates and the world displacement is set to zero. For later image
pairs, the 3D displacement is calculated by subtracting the reference stereo
coordinates from the current triangulated coordinates,

.. math::

   u_X^k = X^k - X^0, \qquad
   u_Y^k = Y^k - Y^0, \qquad
   u_Z^k = Z^k - Z^0.

These are the displacement components written by Pyvale as the stereo
millimetre displacement fields. Subsets that do not pass the temporal or stereo
correlation threshold are not used for the world-coordinate calculation.

Practical Notes
---------------
Stereo DIC is more sensitive to calibration quality than 2D DIC. A poor
intrinsic calibration, stereo rotation, or translation vector will give poor
epipolar lines and therefore poor initial guesses. It will also directly affect
the triangulated world coordinates.

Good speckle texture remains important. The left and right cameras see the same
surface from different viewpoints, so subsets should have enough unique texture
to match reliably even when perspective and lighting change slightly.

The epipolar FFT estimate is used to get close to the correct match, but the
final result still depends on the nonlinear subset optimizer. As with standard
DIC, the threshold, subset size, subset step, shape function, and interpolation
routine all influence the final result.
