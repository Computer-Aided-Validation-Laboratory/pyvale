import numpy as np
import pyvista as pv

num_points = 100
point_cloud = np.random.random((num_points, 3))

# Define a plane
origin = [0, 0, 0]
normal = [0, 0, 1]
plane = pv.Plane(center=origin, direction=normal)


def project_points_to_plane(points, plane_origin, plane_normal):
    """Project points to a plane."""
    vec = points - plane_origin
    dist = np.dot(vec, plane_normal)
    return points - np.outer(dist, plane_normal)


projected_points = project_points_to_plane(point_cloud, origin, normal)

# Create a polydata object with projected points
polydata = pv.PolyData(projected_points)

# Mesh using delaunay_2d and pyvista
mesh = polydata.delaunay_2d()

# Create a plane for visualization
plane_vis = pv.Plane(
    center=origin, direction=normal, i_size=2, j_size=2, i_resolution=10, j_resolution=10
)

# plot it
pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, color='white', opacity=0.5, label='Tessellated mesh')
pl.add_mesh(
    pv.PolyData(point_cloud),
    color='red',
    render_points_as_spheres=True,
    point_size=10,
    label='Points to project',
)
pl.add_mesh(plane_vis, color='blue', opacity=0.1, label='Projection Plane')
pl.add_legend()
pl.show()