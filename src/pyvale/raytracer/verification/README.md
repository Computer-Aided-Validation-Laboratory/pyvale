## Verification case 1: Newton solver reprojection error

Surface position on the tetrahedron's face (weighted sum of shape functions):
$$
	P(g,h) = \sum_{i=0}^{5} N_i(g,h) \cdot f_{nodes_i}
$$
where $g$ and $h$ are element barycentric coordinates.
Ray
$$
R(t) = o + td
$$
where $d$ is the ray's direction, and $o$ is the ray's origin.
Forward map
$$
(g, h) \rightarrow P(u, v) \rightarrow d(u, v)
$$
where
$$
d = \frac{P-o}{|| P-o ||}
$$

Newton solver finds $(g, h)$ such that
$$
R(u, v) = 0
$$

### 1. Generate ground truth samples
Dense grid  $(g_i, h_i)_{true}$
### 2. Calculate true surface locations (forward mapping)
True surface locations
$$
P_{true\_i} = P(g_i, h_i)
$$
And corresponding true ray directions
$$
d_{true\_i} = \frac{P_{true\_i}-o}{|| P_{true\_i}-o ||}
$$
### 3. Recover ground truth samples
Use $d_{true\_i}$  and Newton solver to calculate  $(g_i, h_i)_{rec}$

### 4. Forward map the recovered samples
Recovered surface locations
$$
P_{rec\_i} = P((g_i, h_i)_{rec})
$$
And corresponding recovered ray directions
$$
d_{rec\_i} = \frac{P_{rec\_i}-o}{|| P_{rec\_i}-o ||}
$$
### 5. Calculate projection error
Angular ray error in radians https://uk.mathworks.com/matlabcentral/answers/83812-how-to-calculate-the-angular-error
$$
e_{ray} = cos^{-1}(d_{true} \cdot d_{rec}) = arccos(d_{true} \cdot d_{rec})
$$

### Implementation

Verification case 1 written for [Riley](https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/tree/main) rasteriser in Zig was modified for the above procedure and converted to C++, in order to easily use it with existing ray tracing functions and structures. The output has the same structure as [Riley verification case 1](https://github.com/Computer-Aided-Validation-Laboratory/zraster/blob/cd944b63da1560abb13645a17e6b3245ef78f67d/src/verif_1_solver.zig) output; consequently, the same [Python post-processing functions](https://github.com/Computer-Aided-Validation-Laboratory/zraster/blob/cd944b63da1560abb13645a17e6b3245ef78f67d/scripts/paper_verif_1.py) can be used.

### Compilation
`g++-13 verif_1_solver.cpp cpp/rtrayintersection_extracted.cpp -o case1`