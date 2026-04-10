# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import itertools
import vedo
import vtk
import pandas as pd
from pathlib import Path

from pyvale.raytracer.rtscene import Scene
from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtmesh import RTMesh

# ================================================================================
# SEAM SPLITTER FOR UV UNWRAPPING
# ================================================================================

# Constants for the plotter

# Create instructions for the user. Dicts, so it is easier to modify if needed
INSTRUCTIONS_EDGE_ON = {
    "LMB": "LMB: Select nodes to create the edge (and move mesh freely)",
    "RMB": "RMB: Unselect the last node",
    "S": "S: Save seam points as edge",
    "C": "C: Export seam points to CSV file",
    "D": "D: Exit edge selection mode",
    "Wheel": "Wheel: Zoom in/out",
    "Hold wheel": "Hold wheel: Move sideways",
    "ESC/Q": "Quit seam selector"
}

INSTRUCTIONS_EDGE_OFF = {
    "LMB": "LMB: Move mesh freely",
    "E": "E: Enter edge selection mode",
    "S": "S: Save seam points as edge (if selected)",
    "C": "C: Export seam points to CSV file (if selected)",
    "Wheel": "Wheel: Zoom in/out",
    "Hold wheel": "Hold wheel: Move sideways",
    "ESC/Q": "Quit seam selector"
}

# Create vedo Text2D objects. Separate so we can use different colours for better readability
NEWLINE = '\n' # \escapes are not allowed inside f-strings, so use this to work around that
# Edge selection mode ON
MODE_ON_TEXT = vedo.Text2D("Edge selection mode ON", pos="top-left", c="green7")
EDGE_ON_STRING = f"\n\nControls:\n{NEWLINE.join(f'{instruction}' for instruction in INSTRUCTIONS_EDGE_ON.values())}"
INSTRUCTIONS_ON_TEXT = vedo.Text2D(EDGE_ON_STRING, pos="top-left", s=0.7)
# Edge selection mode OFF
MODE_OFF_TEXT = vedo.Text2D("Edge selection mode OFF", pos="top-left", c="red7")
EDGE_OFF_STRING = f"\n\nControls:\n{NEWLINE.join(f'{instruction}' for instruction in INSTRUCTIONS_EDGE_OFF.values())}"
INSTRUCTIONS_OFF_TEXT = vedo.Text2D(EDGE_OFF_STRING, pos="top-left", s=0.7)

# Layout:
        # plt.at(0) - Small window for text instructions in top left
        # plt.at(1) - Small window displaying mesh as seen by the ray tracer renderer camera
        # plt.at(2) - Bigger window on the left for node/edge selection
# NB4: If/when we get to stereo rendering with 2 cameras at once, this will probably have to be expanded with a conditional statement
# to select appropriate layout based on the number of cameras. For now, it is hardcoded for 1 camera.
PLOTTER_LAYOUT = [{'bottomleft': (0.0, 0.7), 'topright': (0.5, 1.0)},
                          {'bottomleft': (0.0, 0.0), 'topright': (0.5, 0.7)},
                          {'bottomleft': (0.5, 0.0), 'topright': (1.0, 1.0)}]

class SeamSelector:
    """ 
    Class for selecting seams on the mesh. The seams can then be used in the workflow, for example to create a split mesh for better UV mapping.
    
    The user can select nodes to create seams, and save them as edges. The seams are saved as lists of node IDs in the RTMesh object, so they can be used later in the workflow.

    Parameters
    ----------
    rtmesh : RTMesh
        The RTMesh object containing the mesh data.
    renderer_camera : Camera
        The Camera object containing the camera data.
    """
    def __init__(self, rtmesh: RTMesh, renderer_camera):
        self.plotter = vedo.Plotter(shape=PLOTTER_LAYOUT, sharecam=False, title="Seam selector")
        #self.v_mesh = vedo.Mesh([rtmesh.node_coords, rtmesh.connectivity]) # Cannot use that as we lose the grid connections between nodes
        self.v_mesh = vedo.Mesh(rtmesh.pyvista_surface).c("silver").lw(0.1)
        self.rtmesh = rtmesh
        # self.v_mesh.backface_culling(True) # Maybe turn it on for bigger meshes if they take time to move?
        # Picker for selecting nodes
        self.picker = vtk.vtkCellPicker()
        # Variables for displayed and selected objects
        #self.edges = []  # List of selected_node_indices
        self.seams = [] # List of lists of selected node IDs (saved). One list = one seam
        self.temp_seams = [] # List of lists of selected node IDs for the current seam; cleared when seam is saved, but allows for undoing selections before saving.
        self.selected_node_indices = [] # List of selected node indices, used to create the seams. Cleared when seam is saved, but allows for undoing selections before saving.
        self.visual_markers = [] # Stored vedo objects (spheres and lines) to keep track of what is currently displayed as part of the seam selection, so we can remove them when needed
        self.edge_selection_on = False # Flag needed, because it is impossible to remove the callbacks without removing ALL of them, including the default ones in vedo
        # Set the ray tracer renderer camera
        self.set_render_camera(renderer_camera)
        self.set_up()

    def set_render_camera(self, camera: Camera) -> None:
        """
        Sets the same camera view as from the passed renderer camera, so the user can see if the seams will be visible from the ray tracing camera's perspective.
        """
        # 
        self.plotter.screensize = [Camera.image_width, Camera.image_height]
        self.plotter.fov = np.degrees(camera.angle_vertical_view)
        self.plotter.at(1).camera.SetPosition(camera.camera_center)
        self.plotter.at(1).camera.SetFocalPoint(camera.point_camera_target)
        self.plotter.at(1).camera.SetViewUp(camera.vector_view_up)
        self.plotter.at(1).renderer.SetInteractive(False)  # Use VTK renderer to freeze this subplot on the RT renderer view

    def set_up(self):
        """
        Sets up the plotter.
        """
        # Add callbacks
        self.plotter.add_callback('KeyPress', self.toggle_edge_mode)
        self.plotter.add_callback('KeyPress', self.print_on_close)

        # Add text
        self.plotter.at(1).add(vedo.Text2D("Ray tracer camera view", pos="top-left", s=0.7))
        self.plotter.at(2).add(vedo.Text2D("Seam selector", pos="top-left", s=0.7))
        self.plotter.at(0).add(MODE_OFF_TEXT)
        self.plotter.at(0).add(INSTRUCTIONS_OFF_TEXT)

        # Add mesh to the plotter
        self.plotter.at(1).add(self.v_mesh)
        self.plotter.at(2).add(self.v_mesh)
        # Turn lighting off. Otherwise we get flat shading automatically when we use
        self.v_mesh.lighting('off')

        # Set the picker tolerance - particularly important for dense meshes
        self.picker.SetTolerance(0.005)

    def check_line_visibility(self, point_id, point_coords) -> bool:
        """
        Compares the node normal and camera direction to determine if line is visible from the ray tracing camera's perspective.
        NB4: Wrote that, then the issue resolved itself? So keep it here just in case, but not currently in use.
        """
        camera = self.plotter.renderers[1].GetActiveCamera()
        camera_position = np.array(camera.GetPosition())
        normal = self.v_mesh.vertex_normals[point_id] # Find normal
        view_dir = point_coords - camera_position # Find view direction
        # Calculate dot product to check if the selected line should be visible from the renderer camera or not
        dot_prod = np.dot(normal, view_dir)
        if dot_prod < 0:  # Front face
            return True
        return False

    def select_edge(self, event) -> None:
        """
        Detects mouse clicks to find the closest node on the mesh, then draws lines connecting consecutive selected nodes using Dijkstra's shortest path algorithm.
        Unless the path is wrong, there is no need to select all nodes on the edge - their IDs are extracted automatically from the line.
        """
        # Check if we are in the edge selection mode
        if not self.edge_selection_on:
            return
        # Ensure we actually clicked on the mesh and not on the view from ray tracing camera
        if not event.actor or event.actor != self.v_mesh or event.at != 2:
            return

        # Find the index of the closest vertex to the click
        # pt_id = v_mesh.closest_point(event.picked3d, return_point_id=True) # Works for single window, but not subplots
        self.picker.Pick(event.picked2d[0], event.picked2d[1], 0, self.plotter.at(event.at).renderer)
        point_id = self.picker.GetPointId()

        # Try except because picker something returns an impossible id, like 820 for a mesh with 200 nodes.
        # This should help, but I could not consciously reproduce this error, so report if it reappears
        try:
            point_coords = self.v_mesh.points[point_id]
        except IndexError:
            print(f"Point {point_id} not found in mesh")
            return

        if (point_id not in self.selected_node_indices) or (self.selected_node_indices[-1] != point_id):
            # if pt_id not in selected_node_indices: # Expanded to allow creation of closed boundaries
            self.selected_node_indices.append(point_id)
            print(f"Selected node indices after clicking: {self.selected_node_indices}")
            # Highlight the selected node with a red sphere
            node_marker = vedo.Sphere(point_coords, r=self.v_mesh.diagonal_size() / 100, c="red")
            self.visual_markers.append(node_marker)
            self.plotter.at(1).add(node_marker)
            self.plotter.at(2).add(node_marker)

            # If we have at least 2 points, draw the path (the seam) between the last two
            if len(self.selected_node_indices) == 1:
                self.temp_seams.append([point_id])
            elif len(self.selected_node_indices) >= 2:
                start_node = self.selected_node_indices[-2]
                end_node = self.selected_node_indices[-1]

                # Find the shortest path along edges using Dijkstra's shortest path algorithm
                path_coords = self.v_mesh.geodesic(start_node, end_node)
                #print(path_coords) # Debug
                path_node_ids = path_coords.pointdata["VertexIDs"]
                path_node_coords = path_coords.points
                path_vertex_count = len(path_node_ids)
                # Draw the path as a thick red line
                line = vedo.Line(path_node_coords, lw=4, c="red")
                self.visual_markers.append(line)
                self.plotter.at(2).add(line)
                self.plotter.at(1).add(line)
                seam = list()
                # Get all nodes that the line goes through and add them to a seam
                # TO DO: figure out a way to do this better (no nested lists), while maintaining the 
                # ability to unselect nodes
                for node in range(1, len(path_node_ids)):
                    # for node_id in path_node_ids:
                    node_id = path_node_ids[node]
                    seam.append(node_id)
                self.temp_seams.append(seam)
                # Not currently needed as the issue resolved itself, but keep it here just in case.
                # Only display the line and markers if they are visible from the ray tracing camera's perspective
                # if check_line_visibility(point_id, point_coords):
                # self.plotter.at(1).add(line)
                # else:
                # self.plotter.at(1).remove(node_marker)
                print(f"Line added: {path_vertex_count} nodes")
        self.plotter.render()

    def unselect_last_point(self, event) -> None:
        """
        Unselects the last point on the RMB press if the mouse is hovered over any node object to prevent completely accidental undo.
        Removes the associated red sphere marker and line leading to this point (if it exists).
        """
        # Check if we are in the edge selection mode
        if not self.edge_selection_on:
            return
        # Ensure we actually clicked on the node
        if not event.actor or event.actor not in self.visual_markers:
            print("No node selected to unselect")
            return

        # Find the index of the closest vertex to the click
        pt_id = self.v_mesh.closest_point(event.picked3d, return_point_id=True)

        if len(self.visual_markers) > 1:  # We must have at least 2 nodes selected
            self.selected_node_indices.pop()  # Remove the last node ID
            self.plotter.at(1).remove(self.visual_markers[-1])
            self.plotter.at(2).remove(self.visual_markers[-1])
            self.visual_markers.pop()
            self.plotter.at(1).remove(self.visual_markers[-1])
            self.plotter.at(2).remove(self.visual_markers[-1])
            self.visual_markers.pop()
            self.temp_seams.pop()
            #print(f"Temp seams after unselecting: {self.temp_seams}")
        elif len(self.visual_markers) == 1:  # We only have one node selected
            self.selected_node_indices.clear()  # Remove the last node IDe
            self.plotter.at(1).remove(self.visual_markers[-1])
            self.plotter.at(2).remove(self.visual_markers[-1])
            self.visual_markers.pop()
        self.plotter.render()

    def save_seam_points(self, event) -> None:
        """
         Saves the selected seam points as an edge on the S key press. Saved seams cannot be modified.
         NB: Currently, the seam points are saved as a list of lists of node IDs, which is not ideal, but it allows for the ability to undo selections and save multiple seams before exporting
         """
        if event.keypress == "s":
            if len(self.selected_node_indices) < 2:
                print("Not enough seam points selected")
                return
            # Convert list of lists to a single list of node IDs
            # Need to think of a better way to do this instead of having nested loops, but while keeping the abiliy to undo selections...
            one_seam = list()
            for seam in self.temp_seams:
                for node_id in seam:
                    one_seam.append(node_id)
            self.seams.append(one_seam)
            # seams.append(temp_seams.copy())
            random_color = list(
                np.random.random(size=3) * 256)  # Random color to distinguish between different saved edges
            for marker in self.visual_markers:
                marker.c(random_color)
            self.plotter.render()
            print(f"Saved seam points: {self.seams}")
            self.selected_node_indices.clear()
            self.visual_markers.clear()
            self.temp_seams.clear()

    def toggle_edge_mode(self, event) -> None:
        """
        Toggles the edge selection mode.
        When the edge selection mode is on, the user can select nodes to create seams. The instructions are updated accordingly.
        When the edge selection mode is off, the user can move the mesh freely without risking accidentally selecting nodes. The instructions are updated accordingly.
        Unbfortunately, it is not possible to remove the callbacks without removing ALL of them, including the default ones in vedo, so we need to use a flag to check
        if we are in the edge selection mode or not.
        """
        if event.keypress == "e":  # Enter edge selection mode
            self.edge_selection_on = True
            self.plotter.add_callback('LeftButtonPress', self.select_edge)
            self.plotter.add_callback('RightButtonPress', self.unselect_last_point)
            self.plotter.add_callback('KeyPress', self.save_seam_points)
            self.plotter.add_callback('KeyPress', self.export_seams)
            self.plotter.at(0).remove(INSTRUCTIONS_OFF_TEXT)
            self.plotter.at(0).remove(MODE_OFF_TEXT)
            self.plotter.at(0).add(INSTRUCTIONS_ON_TEXT)
            self.plotter.at(0).add(MODE_ON_TEXT)
            self.plotter.render()
        if event.keypress == 'd':  # Leave edge selection mode. Mesh can now be rotated without risking something accidentally
            self.edge_selection_on = False
            self.plotter.at(0).remove(INSTRUCTIONS_ON_TEXT)
            self.plotter.at(0).remove(MODE_ON_TEXT)
            self.plotter.at(0).add(INSTRUCTIONS_OFF_TEXT)
            self.plotter.at(0).add(MODE_OFF_TEXT)
            self.plotter.render()

    def print_on_close(self, event) -> None:
        """
        Prints the final seam node indices on the Q or ESC key press, which also closes the plotter. The seams are saved in the RTMesh object, so they can be used later in the workflow.
        """
        if event.keypress == "q" or event.keypress == "esc":
            self.plotter.close()
            self.rtmesh.seams = self.seams # Save the seams
            print("Final seam node indices:", self.seams)

    def export_seams(self, event) -> None:
        """
        Exports the selected seam node indices to a CSV file on the C key press. The user can specify the filename, but if they leave it empty,
        it will be saved as "seams.csv" in the "pyvale-output" folder in the current working directory.
        """
        if event.keypress == "c":
            if len(self.seams) == 0:
                print("No seams to export")
                return
            base_dir = Path.cwd() / "pyvale-output"
            # Could also do this with tkinter to open file explorer and let the user select; didn't do it to avoid extra dependencies (and not sure how this would work on Linux)
            filename = input("Enter output filename (without the extension) or leave empty to keep it as 'seams.csv': ")
            if filename == "":
                filename = "seams"
            filepath = base_dir.joinpath((filename + ".csv"))
            temp_df = pd.DataFrame(self.seams)
            #for col in temp_df.columns:
            #    temp_df[col] = temp_df[col].astype("Int64")
            temp_df.to_csv(filepath, header=False, index=False) # Use sep=";" if viewing in Excel as otherwise it stacks the data in one cell

    def import_seams_and_run(self, filepath) -> None:
        """
        Reads the seam node indices from a CSV file and visualizes them on the mesh. Additional seams can be added, or user can just verify the existing ones and then
        append them to the RTMesh object for further processing.
        """
        # Read data from csv file
        try:
            temp_df = pd.read_csv(filepath, sep=",", header=None, dtype="Int64") # Pandas turns integers to floats if some rows have NaN for the same columns, so we need to force integers
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return
        temp_df = temp_df.fillna(-1).astype(int) # Replace <NA> with -1
        list_of_seams = temp_df.values.tolist() # Convert to list of lists to match the formatting in the rest of the workflow
        one_seam = list()
        for seam in list_of_seams:
            random_color = list(np.random.random(size=3) * 256)  # Random color to distinguish between different saved edges
            sphere_0 = vedo.Sphere(self.v_mesh.points[seam[0]], r=self.v_mesh.diagonal_size() / 100, c=random_color)
            self.plotter.at(1).add(sphere_0)
            self.plotter.at(2).add(sphere_0)
            one_seam.append(seam[0])
            for i in range(1, len(seam)):
                node_id = seam[i]
                if node_id == -1: # -1 marks start of empty values for that seam (pandas fills it with NaN), so break
                    break
                line = vedo.Line(self.v_mesh.points[seam[i-1]], self.v_mesh.points[node_id], lw=4, c=random_color)
                sphere = vedo.Sphere(self.v_mesh.points[node_id], r=self.v_mesh.diagonal_size() / 100, c=random_color)
                self.plotter.at(1).add(line)
                self.plotter.at(2).add(line)
                self.plotter.at(1).add(sphere)
                self.plotter.at(2).add(sphere)
                one_seam.append(node_id)
            self.seams.append(one_seam.copy())
            one_seam.clear()
        print(f"Imported seams: {self.seams}")
        self.plotter.render()
        self.run()

    def run(self) -> None:
        """
        Runs the plotter.
        """
        self.plotter.show(interactive=True)

# ================================================================================
# MESH UTILS to help in positioning
# ================================================================================

def get_mesh_size(rtmesh: RTMesh, timestep: int = 0) -> np.ndarray:
    """
    Returns the size of the mesh in world units at the given timestep.

    Parameters
    ----------
    rtmesh : RTMesh
        The RTMesh object containing the mesh data.
    timestep : int, optional
        The timestep to get the mesh size for (default is 0 for static meshes).
    Returns
    -------
    np.ndarray
        The size of the mesh in world units at the given timestep.
    """
    coords = rtmesh.node_coords_expanded_over_time[timestep, :, :] # Coords
    spans = np.abs(coords.max(axis=(0,1)) - coords.min(axis=(0,1)))
    return spans

def get_mesh_bounding_box(rtmesh: RTMesh, timestep: int = 0) -> dict:
    """
    Returns mesh position information as a dictionary containing the minimum and maximum x, y, and z coordinates,
    and the center (mean nodal position; not an actual nodal position) in world coordinates at the given timestep.

    Parameters
    ----------
    rtmesh : RTMesh
        The RTMesh object containing the mesh data.
    timestep : int, optional
        The timestep to get the bounding box for (default is 0 for static meshes).
    Returns
    -------
    dict
        A dictionary containing the minimum and maximum x, y, and z coordinates, and the center (mean nodal position; not an actual nodal position) in world coordinates at the given timestep.
    """
    coords = rtmesh.node_coords_expanded_over_time[timestep, :, :]
    #coords = scene.coords_expanded[mesh_idx][timestep, :, :]
    center = coords.mean(axis=(0,1))
    minimal_coords = coords.min(axis=(0,1))
    maximal_coords = coords.max(axis=(0,1))
    return {"min_corner": minimal_coords, "max_corner": maximal_coords, "center": center}

def is_visible_in_viewport(mesh: RTMesh, camera:Camera, timestep: int = 0):
    """
    Checks if a mesh is visible in the viewport of a camera.

    Parameters
    ----------
    mesh : RTMesh
        The RTMesh object containing the mesh data.
    camera : Camera
        The Camera object containing the camera data.
    timestep : int, optional
        The timestep to check the mesh visibility for (default is 0 for static meshes).
    Returns
    -------
    bool
        True if the mesh is fully or partially visible in the viewport, False if the mesh is completely outside the viewport or behind the camera,
        along with hints on how to adjust the mesh position to make it visible, and an estimate of how much of the viewport it should cover if it is visible.
    """
    # Get mesh bounds
    mesh_aabb = get_mesh_bounding_box(mesh, timestep)
    mesh_center = mesh_aabb["center"]
    half_extents = get_mesh_size(mesh, timestep)/2

    # Extract camera basis vectors from the camera-to-world matrix
    c2w = camera.matrix_camera_to_world
    cam_right = c2w[0,:3]
    cam_up = c2w[1, :3]
    cam_forward = -c2w[2, :3]  # Camera looking down -z hence negative

    # Perspective projection parameters
    # tan(FOV/2) gives us the boundary of the frustum at distance 1.0; or, half of the viewport height
    h_temp = np.tan(camera.angle_vertical_view / 2)
    aspect_ratio = camera.image_width / camera.image_height
    tan_half_fov_h = h_temp * aspect_ratio

    # Check the 8 corners of the AABB
    corners_in_front = 0
    any_in_view = False

    # Track NDC bounds to see how much of the screen it covers
    ndc_bounds = {"x": [np.inf, -np.inf], "y": [np.inf, -np.inf]}

    for signs in itertools.product([-1, 1], repeat=3):
        p_world = mesh_center + (np.array(signs) * half_extents)
        delta = p_world - camera.camera_center

        # Project world point onto camera axes (transform to camera space)
        z_cam = np.dot(delta, cam_forward)
        x_cam = np.dot(delta, cam_right)
        y_cam = np.dot(delta, cam_up)

        # Skip points behind the camera (near plane clipping)
        if z_cam <= 0.001:
            continue

        corners_in_front += 1
        # Project to Normalized Device Coordinates (NDC)
        # Range will be [-1, 1] if the point is inside the frustum
        x_ndc = x_cam / (z_cam * h_temp)
        y_ndc = y_cam / (z_cam * h_temp)

        ndc_bounds["x"][0] = min(ndc_bounds["x"][0], x_ndc)
        ndc_bounds["x"][1] = max(ndc_bounds["x"][1], x_ndc)
        ndc_bounds["y"][0] = min(ndc_bounds["y"][0], y_ndc)
        ndc_bounds["y"][1] = max(ndc_bounds["y"][1], y_ndc)

        if -1 <= x_ndc <= 1 and -1 <= y_ndc <= 1:
            any_in_view = True

    if corners_in_front == 0:
        print(f"Mesh is entirely behind the camera.")
        return False

    if not any_in_view:
        # Check if the mesh is so large it spans the whole viewport, but it's corners are outside
        if (ndc_bounds["x"][0] < -1 and ndc_bounds["x"][1] > 1 and
                ndc_bounds["y"][0] < -1 and ndc_bounds["y"][1] > 1):
            print(f"Mesh spans the whole viewport, but its corners are outside.")
            return True
        # Positioning guidance if the mesh is outside the frustum
        hints = []
        # Horizontal positioning
        if ndc_bounds["x"][1] < -1:
            hints.append("too far LEFT (increase X or pan camera left)")
        elif ndc_bounds["x"][0] > 1:
            hints.append("too far RIGHT (decrease X or pan camera right)")

        # Vertical positioning
        if ndc_bounds["y"][1] < -1:
            hints.append("too far DOWN (increase Y or tilt camera down)")
        elif ndc_bounds["y"][0] > 1:
            hints.append("too far UP (decrease Y or tilt camera up)")

        # Depth positioning (if all corners were behind the camera)
        if corners_in_front == 0:
            hints.append("BEHIND the camera (check Z-positioning)")

        direction_str = " and ".join(hints) if hints else "out of frustum range"
        print(f"Mesh is invisible: {direction_str}.")

        # Debugging values
        print(f"   NDC X-range: [{ndc_bounds['x'][0]:.2f}, {ndc_bounds['x'][1]:.2f}]")
        print(f"   NDC Y-range: [{ndc_bounds['y'][0]:.2f}, {ndc_bounds['y'][1]:.2f}]")

        return False

    # Estimate screen coverage - useful for scaling
    # Clamp bounds to screen for area calculation
    x_range = min(1, ndc_bounds["x"][1]) - max(-1, ndc_bounds["x"][0])
    y_range = min(1, ndc_bounds["y"][1]) - max(-1, ndc_bounds["y"][0])

    # Area in NDC is 2x2=4. Percentage of screen:
    pct_coverage = (max(0, x_range) * max(0, y_range) / 4.0) * 100

    print(f"Mesh is fully visible in viewport. It should occupy about {pct_coverage:.2f}% of the viewport.")
    return True