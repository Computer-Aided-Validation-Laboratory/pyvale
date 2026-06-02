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
from scipy.spatial.transform import Rotation

from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtmesh import RTMesh

# ================================================================================
# SCENE VISUALISER
# ================================================================================
SCENE_VIS_FONT_SIZE = 0.6
# Text display for the plotter

# Instructions for mesh snapping
SNAPPING_INSTR_CONST = vedo.Text2D("Mesh snapping ON: \n'RMB': Unselect last point.\n", pos="bottom-left", s=SCENE_VIS_FONT_SIZE) # Constant text
# Below instructions depend on the number of the points selected, so the indices in the tuple conveniently correspond to the length of self.selected_points at which each instruction needs to be displayed
SNAPPING_INSTR = (vedo.Text2D("\n\nSelect the anchor point on the mesh to be moved.", pos="bottom-left", s=SCENE_VIS_FONT_SIZE, c="purple"),
                  vedo.Text2D("\n\nSelect the point on the target mesh to align the anchor point to.", pos="bottom-left", s=SCENE_VIS_FONT_SIZE, c="purple"),
                  vedo.Text2D("\n\nPress 's' to translate the mesh.", pos="bottom-left", s=SCENE_VIS_FONT_SIZE, c="purple"))

# Create instructions for the user. Dict, so it is easier to modify if needed
# All controls displayed to the user
# Nb4 keybindings are what they are to avoid conflicting with the defaults existing in vedo as they cannot be overwritten
INSTRUCTIONS_CAM = {
    "e": "'e': Toggle snapping meshes together ON/OFF",
    "b": "'LMB (click) + b': Make mesh invisible",
    "v": "'v': Show invisible meshes",
    "c": "'c': Toggle camera locators ON/OFF",
    "m": "'m': Reset view to camera POV (if it breaks, just scroll the mouse wheel front and back once.)",
    "ESC": "'ESC': Quit."}

# No camera passed - we ignore the camera-related controls
INSTRUCTIONS_NO_CAM = INSTRUCTIONS_CAM.copy()
INSTRUCTIONS_NO_CAM.pop("c")
INSTRUCTIONS_NO_CAM.pop("m")

NEWLINE = '\n' # \escapes are not allowed inside f-strings, so use this to work around that
NO_CAM_STRING = f"Controls:\n{NEWLINE.join(f'{instruction}' for instruction in INSTRUCTIONS_NO_CAM.values())}"
CAM_STRING = f"Controls:\n{NEWLINE.join(f'{instruction}' for instruction in INSTRUCTIONS_CAM.values())}"
INSTRUCTIONS_NO_CAM_TEXT = vedo.Text2D(NO_CAM_STRING, pos="top-left", s=SCENE_VIS_FONT_SIZE)
INSTRUCTIONS_CAM_TEXT = vedo.Text2D(CAM_STRING, pos="top-left", s=SCENE_VIS_FONT_SIZE)

NEWLINE_LOCATORS = '\n' * (len(INSTRUCTIONS_CAM) + 1) # How many newlines we need to offset the camera locator text to avoid printing it on top of the controls
CAM_POS_TXT = vedo.Text2D(NEWLINE_LOCATORS+ 'Camera position', pos="top-left", s=SCENE_VIS_FONT_SIZE, c="teal")
CAM_LOOKAT_TXT = vedo.Text2D(NEWLINE_LOCATORS +'\nCamera lookat vector', pos="top-left", s=SCENE_VIS_FONT_SIZE, c="indigo")
VIEWPORT_TXT = vedo.Text2D(NEWLINE_LOCATORS +'\n\nViewport', pos="top-left", s=SCENE_VIS_FONT_SIZE, c="plum")
CONE_TXT = vedo.Text2D(NEWLINE_LOCATORS +'\n\n\nDefocus disk/cone', pos="top-left", s=SCENE_VIS_FONT_SIZE, c="turquoise")
CAM_LOC_TXT = [CAM_POS_TXT, CAM_LOOKAT_TXT, VIEWPORT_TXT]

class SceneVisualiser:
    """ 
    Helper displaying the scene in 3D with moveable camera, camera locators (optional), and axes to help orient objects in the world. 
    
    It also allows viewing the scene from the perspective of the passed camera.

    Parameters
    ----------
    rtmeshes : list[RTMesh]
        The list of RTMesh objects in the scene to display.
    renderer_camera : Camera | None
        Optional. The Camera object containing the camera data.
    """
    def __init__(self, rtmeshes: list[RTMesh], renderer_camera: Camera | None = None):
        self.plotter = vedo.Plotter(title="Scene visualiser", axes=1)
        self.rtmeshes: list[RTMesh] | RTMesh = rtmeshes # RTMesh objects displayed
        self.vmeshes: list[vedo.Mesh] = [] # Vedo Mesh objects corresponding to the passed RTMesh objects
        self.invisible_vmeshes: list[vedo.Mesh] = []
        self.camera: Camera | None = renderer_camera
        self.camera_locators: list[vedo.Mesh] = []
        self.display_locators: bool = False # Flag for toggling camera locators on/off
        # Bits for translating meshes and snapping them together
        self.visual_markers: list[vedo.Mesh] = []
        self.selected_points: list = []
        self.affected_meshes: list[int] = [] # Indices of the meshes affected by the snapping. 1st is the mesh to be moved, 2nd is the target
        self.translation_vector: np.ndarray = np.zeros(3) # Translation vector used to move the mesh
        self.snapping_on: bool = False # Flag for toggling mesh snapping on/off
        self._set_up()
        self.plotter.show(interactive=True)
        
    def _snap_back_to_camera_pov(self, event) -> None:
        """
        Callback, which resets the view back to the passed rendered camera view.
        """
        if event.keypress.lower() == "m": 
            self.plotter.screensize = [self.camera.image_width, self.camera.image_height]
            self.plotter.fov = np.degrees(self.camera.angle_vertical_view)
            self.plotter.camera.SetPosition(self.camera.camera_center)
            self.plotter.camera.SetFocalPoint(self.camera.point_camera_target)
            self.plotter.camera.SetViewUp(self.camera.vector_view_up)
            self.plotter.render()

    def _add_camera_locator(self) -> None:
        """
        Creates a blue sphere that represents the camera center position and a red line, representing the camera lookat vector.
        """
        # Rectangle that will show there the camera is located
        cam_size = self.vmeshes[0].diagonal_size() / 50 # Size of the camera is based on the diagonal size of the first mesh passed to the class
        camera_vmesh = vedo.Star3D(pos = self.camera.camera_center, r=cam_size, c="teal", alpha = 0.6)
        self.plotter.add(camera_vmesh)
        lookat_width = cam_size/10
        camera_lookat = vedo.Arrow2D(self.camera.camera_center, self.camera.point_camera_target, shaft_width = lookat_width, head_width = 6*lookat_width, c="indigo")
        self.plotter.add(camera_lookat)
        # Vedo asks for bottom left and top right corners, so we flip the x-axis to display the viewport correctly
        viewport = vedo.Rectangle(self.camera.viewport_bottom_right[:1]*np.array([-1, 1]), self.camera.viewport_upper_left[:1]*np.array([-1, 1]), c="plum", alpha = 0.4)
        # Now we offset the z-location as by default, the rectangle is created at the origin which may not be correct
        # (You can pass a 3D array to vedo.Rectangle, but it created the rectangle incorrectly: while the specified corners had correct positions, the remaining two were always at z=0, so it was bent)
        viewport.pos([0,0,self.camera.viewport_bottom_right[2]] + viewport.transform.position)
        self.plotter.add(viewport)
        self.camera_locators.append(viewport)
        self.camera_locators.append(camera_vmesh)
        self.camera_locators.append(camera_lookat)
        # Show cone for DoF in thin lens approximation if applicable
        cone_tan = np.tan(self.camera.angle_cone / 2)
        radius_defocus_disc = self.camera.focal_length * cone_tan
        if radius_defocus_disc > 0:
            # Find the axis around which the cone is oriented; it's equivalent to the camera basis forward vector. Must be normalised
            axis = self.camera.matrix_camera_to_world[2, :3] # 3rd row in camera_to_world_matrix = normalised basis forward vector
            v = vedo.utils.versor(axis) * self.camera.focal_length / 2 # This is what vedo uses to offset the base and top, so we reverse engineer to find the center
            focus_cone = vedo.Cone(pos=self.camera.camera_center - v, r=radius_defocus_disc, height = self.camera.focal_length, c="turquoise", alpha = 0.15)
            # Unfortunately, vedo cone always starts so that base_z < top_z and we want the apex to be at the viewport, so we must flip it around the basis vector up
            focus_cone.rotate(180, axis = self.camera.vector_view_up, point = self.camera.camera_center - v)
            self.plotter.add(focus_cone)
            self.camera_locators.append(focus_cone)
            CAM_LOC_TXT.append(CONE_TXT)
        self.display_locators = True # Toggle flag

    def _toggle_camera_locators(self, event) -> None:
        """
        Toggles the visibility of camera locators in the scene.
        """
        if event.keypress.lower() == "c":
            if self.display_locators: # If we display those, then we know that the list is not empty so we don't check that
                self.plotter.remove(self.camera_locators[0])
                self.plotter.remove(self.camera_locators[1])
                self.plotter.remove(CAM_LOC_TXT)
            else:
                self.plotter.add(self.camera_locators)
                self.plotter.add(CAM_LOC_TXT)
            self.display_locators = not self.display_locators # Set flat to its opposite
            self.plotter.render() # Update the plot
    
    def _add_rtmesh(self, rtmesh: RTMesh) -> None:
        """
        Helper function handling all data processing to add an RTMesh object to the SceneVisualiser.

        Parameters
        ----------
        rtmesh : RTMesh
            The RTMesh object to add to the scene.
        """
        if not isinstance(rtmesh, RTMesh):
            raise TypeError("Meshes must be of type RTMesh")
        v_mesh = vedo.Mesh(rtmesh.pyvista_surface).c("silver").lw(0.1)
        self.vmeshes.append(v_mesh) # List of Vedo meshes, so we can always erase the scene if we want to
        self.plotter.add(v_mesh)
        v_mesh.lighting('off')

    def _set_up(self) -> None:
        """
        Sets up the plotter by displaying instructions, adding callbacks, and setting up the camera if applicable.
        """
        if isinstance(self.rtmeshes, list):
            for rtmesh in self.rtmeshes:
                self._add_rtmesh(rtmesh)
        else:
            self._add_rtmesh(self.rtmeshes)
        
        # Debug
        #for idx, mesh in enumerate(self.vmeshes):
        #    bound = mesh.bounds()
        #    x_size = bound[1] - bound[0]
        #    y_size = bound[3] - bound[2]
        #    z_size = bound[5] - bound[4]
        #    print(f"Mesh {idx} bounds. x: {x_size}, y: {y_size}, z: {z_size}")


        # Add general callbacks
        self.plotter.add_callback('KeyPress', self._snap_meshes)
        self.plotter.add_callback('LeftButtonPress', self._select_point)
        self.plotter.add_callback('RightButtonPress', self._unselect_last_point)
        self.plotter.add_callback('KeyPress', self._translate_mesh)
        self.plotter.add_callback('KeyPress', self._toggle_visibility)

         # Add camera-specific functions and callbacks
        if self.camera is not None:
            self.plotter.add_callback('KeyPress', self._snap_back_to_camera_pov)
            self.plotter.add_callback('KeyPress', self._toggle_camera_locators)
            self._add_camera_locator()
            self.plotter.add(INSTRUCTIONS_CAM_TEXT) 
            self.plotter.add(CAM_LOC_TXT)
            
        else:
            self.plotter.add(INSTRUCTIONS_NO_CAM_TEXT)

    def _adjust_snapping_instructions(self, add = True) -> None:
        """
        Updates the displayed instructions for the user for mesh snapping depending on the number of selected points.

        Parameters
        ----------
        add: bool
            Specifies if points are being selected or unselected, as otherwise it does not work properly for unselecting points. Defaults to True.
        """
        selected_pts_count = len(self.selected_points)

        # If unselecting points, we need to remove the current instruction and replace it with the previous one
        # Unselecting function checks if a valid point was selected, which prevents us from trying to index sub-0
        if not add:
            self.plotter.remove(SNAPPING_INSTR[selected_pts_count+1])
        else:
            # Remove the instruction for the previous number of points
            self.plotter.remove(SNAPPING_INSTR[selected_pts_count-1])

        if self.snapping_on:
            # If snapping is on, add the instructionc corresponding to the current number of selected points
            self.plotter.add(SNAPPING_INSTR[selected_pts_count])
        else:
           self.plotter.remove(SNAPPING_INSTR[selected_pts_count]) # Remove the first instruction if snapping is toggled off

    def _snap_meshes(self, event) -> None:
        """
        Switches the mesh snapping mode on/off and displays appropriate instructions.
        """
        if event.keypress.lower() == "e":
            self.snapping_on = not self.snapping_on
            print("Snapping meshes ON/OFF: ", self.snapping_on)
            if self.snapping_on:
                self.plotter.add(SNAPPING_INSTR_CONST)
                self.plotter.add(SNAPPING_INSTR[0]) # Add instruction about selecting the first point
            else:
                self.plotter.remove(SNAPPING_INSTR_CONST)
                self._adjust_snapping_instructions() # Remove instructions
            self.plotter.render()

    def _select_point(self, event) -> None:
        """
        Detects mouse clicks to find the closest node on the mesh, then draws lines connecting consecutive selected nodes using Dijkstra's shortest path algorithm.
        Unless the path is wrong, there is no need to select all nodes on the edge - their IDs are extracted automatically from the line.
        """
        # Check if we are in the mesh snapping mode
        if not self.snapping_on:
            return
        # Ensure we actually clicked on one of the meshes in the scene and not a marker/something else
        if not event.actor or event.actor not in self.vmeshes:
            return
        
        if len(self.selected_points) == 2:
            print("You only need 2 points to snap the meshes, so you cannot select more.")

        v_mesh = event.actor # Vedo mesh we clicked on
        # Find the index of the closest vertex to the click
        point_id = v_mesh.closest_point(event.picked3d, return_point_id=True) # Works for single window, but not subplots

        # Try except because picker something returns an impossible id, like 820 for a mesh with 200 nodes.
        try:
            point_coords = v_mesh.points[point_id]
        except IndexError:
            print(f"Point {point_id} not found in mesh")
            return

        # If there are selected points, check validity first
        if len(self.selected_points) > 0:
            # Equivalent of "if point_coords not in self.selected_points", but working for numpy arrays. Nb4 this requires all of them to have the same shape, which is the case here
            # Unfortunately, this means we can't run this check without confirming that self.selected_points isn't empty or numpy will throw an error
            if np.any(np.all(point_coords == self.selected_points, axis=1)): 
                print("Reselected the same point.")
                return
            if v_mesh in self.affected_meshes:
                print("You cannot snap a mesh to itself.")
                return
            
        self.selected_points.append(point_coords)
        self.affected_meshes.append(v_mesh)
        # Highlight the selected node with a red sphere
        node_marker = vedo.Sphere(point_coords, r=v_mesh.diagonal_size() / 100, c="pink")
        self.visual_markers.append(node_marker)
        self.plotter.add(node_marker)

        self._adjust_snapping_instructions() # Change instructions based on how many points we have selected

        if len(self.selected_points) == 2:
            self.translation_vector = self.selected_points[1] - self.selected_points[0]
            line = vedo.Line(self.selected_points[0], self.selected_points[1], lw=3, c="pink")
            self.visual_markers.append(line)
            self.plotter.add(line)
        self.plotter.render()

    def _unselect_last_point(self, event) -> None:
        """
        Unselects the last point on the RMB press if the mouse is hovered over any node object to prevent completely accidental undo.
        Removes the associated sphere marker and line leading to this point (if it exists).
        """
        # Check if we are in the snapping mode where we can select points
        if not self.snapping_on:
            return
        # Ensure we actually clicked on the node
        if not event.actor or event.actor not in self.visual_markers:
            print("No node selected to unselect")
            return

        if len(self.visual_markers) > 1:  # We must have at least 2 nodes selected
            self.selected_points.pop()  # Remove the last point
            self.affected_meshes.pop()
            self.plotter.remove(self.visual_markers[-1])
            self.visual_markers.pop()
            # Remove LINE that connected this node to the previous one
            self.plotter.remove(self.visual_markers[-1])
            self.visual_markers.pop()
        elif len(self.visual_markers) == 1:  # We only have one node selected => No lines
            self.selected_points.clear() 
            self.plotter.remove(self.visual_markers[-1])
            self.visual_markers.pop()
        self._adjust_snapping_instructions(add=False)
        self.plotter.render()

    def _reset_snapping(self) -> None:
        """
        Resets the attributes related to snapping meshes.
        """
        self.selected_points.clear() 
        self.affected_meshes.clear()
        self.translation_vector = np.zeros(3)
        for marker in self.visual_markers:
            self.plotter.remove(marker)
        self.visual_markers.clear()
        self._adjust_snapping_instructions()
    

    def _translate_mesh(self, event) -> None:
        """
        Translates the selected RTMesh by the vector determined by the points selected by the user.
        """
        if not self.snapping_on or len(self.affected_meshes) < 2:
            return
        if event.keypress.lower() == "s":
            moved_mesh_idx = self.vmeshes.index(self.affected_meshes[0]) # Find index of the mesh that we want to move
            # Translate the RTMesh
            print(f"Translation vector for RTMesh with index {moved_mesh_idx}: {self.translation_vector}") # Print, because more often than not we will want to make note of this vector and use it to programatically translate the mesh for good
            self.rtmeshes[moved_mesh_idx].translate(self.translation_vector)
            # This updates the pyvista surface used to create the vedo vmesh, so we need to update it
            new_vmesh = vedo.Mesh(self.rtmeshes[moved_mesh_idx].pyvista_surface).c("silver").lw(0.1)
            self.plotter.remove(self.vmeshes[moved_mesh_idx]) # Remove old vedo mesh
            self.plotter.add(new_vmesh) # Add new vedo mesh to the plotter
            self.vmeshes[moved_mesh_idx] = new_vmesh # Replace with new vedo mesh
            # Clean the visualiser
            self._reset_snapping()
            self.plotter.render()

    def _toggle_visibility(self, event) -> None:
        """
        Toggles the visibility of meshes in the scene. Vedo comes with keybind to make a mesh invisible by pressing "x", but there is no way to make it visible again
        if you click elsewhere, so this function presents a workaround.

        It might be helpful with selecting points for snapping meshes if the space between the two is tight already.
        """
        if event.keypress.lower() == "b":
            if event.actor in self.vmeshes and event.actor not in self.invisible_vmeshes:
                # We don't remove it from self.vmeshes as we want to preserve the mesh order there to match the indexing in self.rtmeshes in case the user wants to move it 
                self.invisible_vmeshes.append(event.actor)
                self.plotter.remove(event.actor)
        if event.keypress.lower() == "v":
            mesh_count = len(self.invisible_vmeshes)
            while (mesh_count > 0):
                vmesh = self.invisible_vmeshes[-1]
                if vmesh not in self.plotter.get_meshes(): # To make sure it is gone for good and we need to add it back to the plotter like this
                    self.plotter.add(vmesh)
                self.invisible_vmeshes.pop()
                mesh_count -= 1
        self.plotter.render()

    def add_mesh(self, rtmesh: RTMesh | list[RTMesh]) -> None:
        """
        Allows user to pass an RTMesh on top of those already existing in the scene.

        Parameters
        ----------
        rtmesh : RTMesh | list[RTMesh]
            Single or multiple RTMesh objects to add to the scene.
        """
        if isinstance(rtmesh, list): # Process list
            for rtmesh in rtmesh:
                self._add_rtmesh(rtmesh)
        else: # Process single mesh
            self._add_rtmesh(rtmesh)
        self.plotter.render()

    def clear_visualiser(self) -> None:
        """
        Allows user to remove all meshes from the current display (but keeps the camera data).
        """
        self.rtmeshes.clear()
        self.vmeshes.clear()
        self.camera_locators.clear()
        self.visual_markers.clear()
        self.selected_points.clear()
        self.affected_meshes.clear()
        self.translation_vector = np.zeros(3)
        self.snapping_on = False
        self.display_locators = False
        self.plotter.clear()
        self._set_up()
        self.plotter.render() # Reset view

# ================================================================================
# SEAM SELECTOR/SPLITTER FOR UV UNWRAPPING
# ================================================================================

# Constants for the plotter

# Create instructions for the user. Dicts, so it is easier to modify if needed
INSTRUCTIONS_EDGE_ON = {
    "LMB": "LMB: Select nodes to create the edge (and move mesh freely)",
    "RMB": "RMB: Unselect the last node",
    "s": "s: Save seam points as edge",
    "c": "c: Export seam points to CSV file",
    "d": "d: Exit edge selection mode",
    "Wheel": "Wheel: Zoom in/out",
    "Hold wheel": "Hold wheel: Move sideways",
    "Q": "Quit seam selector"
}

INSTRUCTIONS_EDGE_OFF = {
    "LMB": "LMB: Move mesh freely",
    "e": "e: Enter edge selection mode",
    "s": "s: Save seam points as edge (if selected)",
    "c": "c: Export seam points to CSV file (if selected)",
    "Wheel": "Wheel: Zoom in/out",
    "Hold wheel": "Hold wheel: Move sideways",
    "Q": "Quit seam selector"
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
        self.plotter: vedo.Plotter = vedo.Plotter(shape=PLOTTER_LAYOUT, sharecam=False, title="Seam selector")
        #self.v_mesh = vedo.Mesh([rtmesh.node_coords, rtmesh.connectivity]) # Cannot use that as we lose the grid connections between nodes
        self.v_mesh: vedo.Mesh = vedo.Mesh(rtmesh.pyvista_surface).c("silver").lw(0.1)
        self.rtmesh: RTMesh = rtmesh
        # self.v_mesh.backface_culling(True) # Maybe turn it on for bigger meshes if they take time to move?
        # Picker for selecting nodes
        self.picker: vtk.vtkCellPicker = vtk.vtkCellPicker()
        # Variables for displayed and selected objects
        #self.edges = []  # List of selected_node_indices
        self.seams: list[list[int]] = [] # List of lists of selected node IDs (saved). One list = one seam
        self.temp_seams: list[list[int]] = [] # List of lists of selected node IDs for the current seam; cleared when seam is saved, but allows for undoing selections before saving.
        self.selected_node_indices = [] # List of selected node indices, used to create the seams. Cleared when seam is saved, but allows for undoing selections before saving.
        self.visual_markers: list[vedo.Mesh] = [] # Stored vedo objects (spheres and lines) to keep track of what is currently displayed as part of the seam selection, so we can remove them when needed
        self.edge_selection_on: bool = False # Flag needed, because it is impossible to remove the callbacks without removing ALL of them, including the default ones in vedo
        # Set the ray tracer renderer camera
        self._set_render_camera(renderer_camera)
        self._set_up()

    def _set_render_camera(self, camera: Camera) -> None:
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

    def _set_up(self):
        """
        Sets up the plotter.
        """
        # Add callbacks
        self.plotter.add_callback('KeyPress', self._toggle_edge_mode)
        self.plotter.add_callback('KeyPress', self._print_on_close)

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

    def _check_line_visibility(self, point_id, point_coords) -> bool:
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

    def _select_edge(self, event) -> None:
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

        # Try except because picker sometimes returns an impossible id, like 820 for a mesh with 200 nodes.
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
            #if len(self.selected_node_indices) == 1:
                #self.temp_seams.append([point_id])
            #elif len(self.selected_node_indices) >= 2:
            if len(self.selected_node_indices) >= 2:
                start_node = self.selected_node_indices[-2]
                end_node = self.selected_node_indices[-1]

                # Find the shortest path along edges using Dijkstra's shortest path algorithm
                path_coords = self.v_mesh.geodesic(start_node, end_node)
                #print(path_coords) # Debug
                path_node_ids = path_coords.pointdata["VertexIDs"]
                path_node_coords = path_coords.points
                #path_vertex_count = len(path_node_ids)
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
                # if _check_line_visibility(point_id, point_coords):
                # self.plotter.at(1).add(line)
                # else:
                # self.plotter.at(1).remove(node_marker)
                #print(f"Line added: {path_vertex_count} nodes")
        self.plotter.render()

    def _unselect_last_point(self, event) -> None:
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

        if len(self.visual_markers) > 1:  # We must have at least 2 nodes selected
            self.selected_node_indices.pop()  # Remove the last node ID
            self.plotter.at(1).remove(self.visual_markers[-1])
            self.plotter.at(2).remove(self.visual_markers[-1])
            self.visual_markers.pop()
            # Remove LINE that connected this node to the previous one
            self.plotter.at(1).remove(self.visual_markers[-1])
            self.plotter.at(2).remove(self.visual_markers[-1])
            self.visual_markers.pop()
            self.temp_seams.pop()
            #print(f"Temp seams after unselecting: {self.temp_seams}")
        elif len(self.visual_markers) == 1:  # We only have one node selected => No lines
            self.selected_node_indices.clear() 
            self.plotter.at(1).remove(self.visual_markers[-1])
            self.plotter.at(2).remove(self.visual_markers[-1])
            self.visual_markers.pop()
        self.plotter.render()

    def _save_seam_points(self, event) -> None:
        """
         Saves the selected seam points as an edge on the S key press. Saved seams cannot be modified.
         NB: Currently, the seam points are saved as a list of lists of node IDs, which is not ideal, but it allows for the ability to undo selections and save multiple seams before exporting
         """
        if event.keypress.lower() == "s":
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

    def _toggle_edge_mode(self, event) -> None:
        """
        Toggles the edge selection mode.
        When the edge selection mode is on, the user can select nodes to create seams. The instructions are updated accordingly.
        When the edge selection mode is off, the user can move the mesh freely without risking accidentally selecting nodes. The instructions are updated accordingly.
        Unbfortunately, it is not possible to remove the callbacks without removing ALL of them, including the default ones in vedo, so we need to use a flag to check
        if we are in the edge selection mode or not.
        """
        if event.keypress.lower() == "e":  # Enter edge selection mode
            self.edge_selection_on = True
            self.plotter.add_callback('LeftButtonPress', self._select_edge)
            self.plotter.add_callback('RightButtonPress', self._unselect_last_point)
            self.plotter.add_callback('KeyPress', self._save_seam_points)
            self.plotter.add_callback('KeyPress', self._export_seams)
            self.plotter.at(0).remove(INSTRUCTIONS_OFF_TEXT)
            self.plotter.at(0).remove(MODE_OFF_TEXT)
            self.plotter.at(0).add(INSTRUCTIONS_ON_TEXT)
            self.plotter.at(0).add(MODE_ON_TEXT)
            self.plotter.render()
        if event.keypress.lower() == 'd':  # Leave edge selection mode. Mesh can now be rotated without risking something accidentally
            self.edge_selection_on = False
            self.plotter.at(0).remove(INSTRUCTIONS_ON_TEXT)
            self.plotter.at(0).remove(MODE_ON_TEXT)
            self.plotter.at(0).add(INSTRUCTIONS_OFF_TEXT)
            self.plotter.at(0).add(MODE_OFF_TEXT)
            self.plotter.render()

    def _print_on_close(self, event) -> None:
        """
        Prints the final seam node indices on the Q or ESC key press, which also closes the plotter. The seams are saved in the RTMesh object, so they can be used later in the workflow.
        """
        if event.keypress.lower() == "q" or event.keypress.lower() == "esc":
            self.plotter.close()
            self.rtmesh.seams = self.seams # Save the seams
            print("Final seam node indices:", self.seams)

    def _export_seams(self, event) -> None:
        """
        Exports the selected seam node indices to a CSV file on the C key press. The user can specify the filename, but if they leave it empty,
        it will be saved as "seams.csv" in the "pyvale-output" folder in the current working directory.
        """
        if event.keypress.lower() == "c":
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

    def import_seams_and_run(self, filepath: str) -> None:
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