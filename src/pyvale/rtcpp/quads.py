#!/usr/bin/env python3

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)
from vtkmodules.vtkCommonDataModel import (
    vtkBiQuadraticQuad,
    vtkBiQuadraticQuadraticHexahedron,
    vtkBiQuadraticQuadraticWedge,
    vtkBiQuadraticTriangle,
    vtkCubicLine,
    vtkQuadraticEdge,
    vtkQuadraticHexahedron,
    vtkQuadraticLinearQuad,
    vtkQuadraticLinearWedge,
    vtkQuadraticPolygon,
    vtkQuadraticPyramid,
    vtkQuadraticQuad,
    vtkQuadraticTetra,
    vtkQuadraticTriangle,
    vtkQuadraticWedge,
    vtkTriQuadraticHexahedron,
    vtkUnstructuredGrid
)
# noinspection PyUnresolvedReferences
from vtkmodules.vtkCommonTransforms import vtkTransform
# noinspection PyUnresolvedReferences
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter
from vtkmodules.vtkFiltersSources import (
    vtkCubeSource,
    vtkSphereSource
)
from vtkmodules.vtkInteractionWidgets import (
    vtkCameraOrientationWidget,
    vtkOrientationMarkerWidget
)
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkDataSetMapper,
    vtkGlyph3DMapper,
    vtkLightKit,
    vtkPolyDataMapper,
    vtkProperty,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTextMapper,
    vtkTextProperty
)
from vtkmodules.vtkRenderingLabel import vtkLabeledDataMapper


def get_program_parameters():
    import argparse
    description = 'Demonstrate the isoparametric cell types found in VTK.'
    epilogue = '''
         The numbers define the ordering of the points making the cell.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('-w', '--wireframe', action='store_true',
                        help='Render a wireframe.')
    group1.add_argument('-b', '--backface', action='store_true',
                        help='Display the back face in a different colour.')

    parser.add_argument('-o', '--object_number', type=int, default=None,
                        help='The number corresponding to the object.')
    parser.add_argument('-n', '--no_plinth', action='store_true',
                        help='Remove the plinth.')
    args = parser.parse_args()
    return args.wireframe, args.backface, args.object_number, args.no_plinth


def main():
    wireframe_on, backface_on, object_num, plinth_off = get_program_parameters()

    objects = specify_objects()
    # The order here should match the order in specify_objects().
    object_order = list(objects.keys())

    # Check for a single object.
    single_object = None
    if object_num:
        if object_num in object_order:
            single_object = True
        else:
            print('Object not found.\nPlease enter the number corresponding to the object.')
            print('Available objects are:')
            for obj in object_order:
                print(f'{objects[obj]} (={str(obj)})')
            return

    colors = vtkNamedColors()

    # Create one sphere for all.
    sphere = vtkSphereSource()
    sphere.SetPhiResolution(21)
    sphere.SetThetaResolution(21)
    sphere.SetRadius(0.04)

    cells = get_unstructured_grids()
    # The text to be displayed in the viewport.
    names = list()
    # The keys of the objects selected for display.
    keys = list()
    if single_object:
        names.append(f'{objects[object_num]} (={str(object_num)})')
        keys.append(object_num)
    else:
        for obj in object_order:
            names.append(f'{objects[obj]} (={str(obj)})')
            keys.append(obj)

    add_plinth = (24, 25, 12, 26, 27, 29, 31, 32, 33)
    lines = (21, 35)

    # Set up the viewports.
    grid_column_dimensions = 4
    grid_row_dimensions = 4
    renderer_size = 300
    if single_object:
        grid_column_dimensions = 1
        grid_row_dimensions = 1
        renderer_size = 1200
    window_size = (grid_column_dimensions * renderer_size, grid_row_dimensions * renderer_size)

    viewports = dict()
    blank = len(cells)
    blank_viewports = list()

    for row in range(0, grid_row_dimensions):
        for col in range(0, grid_column_dimensions):
            index = row * grid_column_dimensions + col
            # Set the renderer's viewport dimensions (xmin, ymin, xmax, ymax) within the render window.
            # Note that for the Y values, we need to subtract the row index from grid_rows
            #  because the viewport Y axis points upwards, and we want to draw the grid from top to down.
            viewport = (float(col) / grid_column_dimensions,
                        float(grid_row_dimensions - (row + 1)) / grid_row_dimensions,
                        float(col + 1) / grid_column_dimensions,
                        float(grid_row_dimensions - row) / grid_row_dimensions)

            if index < blank:
                viewports[keys[index]] = viewport
            else:
                s = f'vp_{col:d}_{row:d}'
                viewports[s] = viewport
                blank_viewports.append(s)

    ren_win = vtkRenderWindow()
    ren_win.SetSize(window_size)
    ren_win.SetWindowName('IsoparametricCellsDemo')

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    # Since we always import vtkmodules.vtkInteractionStyle we can do this
    # because vtkInteractorStyleSwitch is automatically imported:
    iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

    renderers = dict()

    # Create and link the mappers, actors and renderers together.
    single_object_key = None
    for idx, key in enumerate(keys):
        print('Creating:', names[idx])

        if single_object:
            single_object_key = key

        text_property = get_text_property()
        if single_object:
            text_property.SetFontSize(renderer_size // 28)
        else:
            text_property.SetFontSize(renderer_size // 24)

        text_mapper = vtkTextMapper()
        text_mapper.SetTextProperty(text_property)
        text_mapper.SetInput(names[idx])
        text_actor = vtkActor2D()
        text_actor.SetMapper(text_mapper)
        text_actor.SetPosition(renderer_size / 2.0, 8)

        mapper = vtkDataSetMapper()
        mapper.SetInputData(cells[key][0])
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.SetProperty(get_actor_property())

        if wireframe_on or key in lines:
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(2)
            actor.GetProperty().SetOpacity(1)
            actor.GetProperty().SetColor(colors.GetColor3d('Black'))
        else:
            if backface_on:
                actor.SetBackfaceProperty(get_back_face_property())

        # Label the points.
        label_property = get_label_property()
        if single_object:
            label_property.SetFontSize(renderer_size // 36)
        else:
            label_property.SetFontSize(renderer_size // 16)

        label_mapper = vtkLabeledDataMapper()
        label_mapper.SetInputData(cells[key][0])
        label_mapper.SetLabelTextProperty(label_property)

        label_actor = vtkActor2D()
        label_actor.SetMapper(label_mapper)

        # Glyph the points.
        point_mapper = vtkGlyph3DMapper()
        point_mapper.SetInputData(cells[key][0])
        point_mapper.SetSourceConnection(sphere.GetOutputPort())
        point_mapper.ScalingOn()
        point_mapper.ScalarVisibilityOff()

        point_actor = vtkActor()
        point_actor.SetMapper(point_mapper)
        point_actor.SetProperty(get_point_actor_property())

        renderer = vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('LightSteelBlue'))
        renderer.SetViewport(viewports[key])

        light_kit = vtkLightKit()
        light_kit.AddLightsToRenderer(renderer)

        renderer.AddActor(text_actor)
        renderer.AddActor(actor)
        renderer.AddActor(label_actor)
        renderer.AddActor(point_actor)
        if not plinth_off:
            # Add a plinth.
            if key in add_plinth:
                tile_actor = make_tile(cells[key][0].GetBounds(),
                                       expansion_factor=0.5, thickness_ratio=0.01, shift_y=-0.05)
                tile_actor.SetProperty(get_tile_property())
                renderer.AddActor(tile_actor)

        renderer.ResetCamera()
        renderer.GetActiveCamera().Azimuth(cells[key][1])
        renderer.GetActiveCamera().Elevation(cells[key][2])
        renderer.GetActiveCamera().Dolly(cells[key][3])
        renderer.ResetCameraClippingRange()

        renderers[key] = renderer
        ren_win.AddRenderer(renderers[key])

    for name in blank_viewports:
        viewport = viewports[name]
        renderer = vtkRenderer()
        renderer.SetBackground = colors.GetColor3d('LightSteelBlue')
        renderer.SetViewport(viewport)

        renderers[name] = renderer
        ren_win.AddRenderer(renderers[name])

    if single_object:
        if vtk_version_ok(9, 0, 20210718):
            try:
                cam_orient_manipulator = vtkCameraOrientationWidget()
                cam_orient_manipulator.SetParentRenderer(renderers[single_object_key])
                cam_orient_manipulator.SetInteractor(iren)
                # Enable the widget.
                cam_orient_manipulator.On()
            except AttributeError:
                pass
        else:
            axes = vtkAxesActor()
            widget = vtkOrientationMarkerWidget()
            rgba = [0.0, 0.0, 0.0, 0.0]
            colors.GetColor("Carrot", rgba)
            widget.SetOutlineColor(rgba[0], rgba[1], rgba[2])
            widget.SetOrientationMarker(axes)
            widget.SetInteractor(iren)
            widget.SetViewport(0.0, 0.0, 0.2, 0.2)
            widget.EnabledOn()
            widget.InteractiveOn()

    ren_win.Render()
    iren.Initialize()
    iren.Start()


def specify_objects():
    """
    Link the unstructured grid number to the unstructured grid name.

    :return: A dictionary: {index number: unstructured grid name}.
    """
    objects = {
        21: 'VTK_QUADRATIC_EDGE',
        22: 'VTK_QUADRATIC_TRIANGLE',
        23: 'VTK_QUADRATIC_QUAD',
        # 36: 'VTK_QUADRATIC_POLYGON',
        24: 'VTK_QUADRATIC_TETRA',
        25: 'VTK_QUADRATIC_HEXAHEDRON',
        26: 'VTK_QUADRATIC_WEDGE',
        # 27: 'VTK_QUADRATIC_PYRAMID',
        28: 'VTK_BIQUADRATIC_QUAD',
        29: 'VTK_TRIQUADRATIC_HEXAHEDRON',
        30: 'VTK_QUADRATIC_LINEAR_QUAD',
        31: 'VTK_QUADRATIC_LINEAR_WEDGE',
        32: 'VTK_BIQUADRATIC_QUADRATIC_WEDGE',
        33: 'VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON',
        34: 'VTK_BIQUADRATIC_TRIANGLE',
        35: 'VTK_CUBIC_LINE',
    }
    return objects


def get_unstructured_grids():
    """
    Get the unstructured grid names, the unstructured grid and initial orientations.

    Get the unstructured grid names, the unstructured grid and initial orientations.

    :return: A dictionary: {index number: (unstructured grid, azimuth, elevation and dolly)}.
    """

    return {
        21: (make_ug(vtkQuadraticEdge()), 0, 0, 0.8),
        22: (make_ug(vtkQuadraticTriangle()), 0, 0, 0),
        23: (make_ug(vtkQuadraticQuad()), 0, 0, 0),
        # 36: (make_quadratic_polygon(), 0, 0, 0),
        24: (make_ug(vtkQuadraticTetra()), 20, 20, 1.0),
        25: (make_ug(vtkQuadraticHexahedron()), -30, 12, 0.95),
        26: (make_ug(vtkQuadraticWedge()), 45, 15, 1.0),
        # 27: (make_quadratic_pyramid(), -110, 8, 1.0),
        28: (make_ug(vtkBiQuadraticQuad()), 0, 0, 0),
        29: (make_ug(vtkTriQuadraticHexahedron()), -15, 15, 0.95),
        30: (make_ug(vtkQuadraticLinearQuad()), 0, 0, 0),
        31: (make_ug(vtkQuadraticLinearWedge()), 60, 22.5, 1.0),
        32: (make_ug(vtkBiQuadraticQuadraticWedge()), 70, 22.5, 1.0),
        33: (make_ug(vtkBiQuadraticQuadraticHexahedron()), -15, 15, 0.95),
        34: (make_ug(vtkBiQuadraticTriangle()), 0, 0, 0),
        35: (make_ug(vtkCubicLine()), 0, 0, 0.85),
    }


# These functions return a vtkUnstructured grid corresponding to the object.

def make_ug(cell):
    pcoords = cell.GetParametricCoords()
    for i in range(0, cell.GetNumberOfPoints()):
        cell.GetPointIds().SetId(i, i)
        cell.GetPoints().SetPoint(i, (pcoords[3 * i]), (pcoords[3 * i + 1]), (pcoords[3 * i + 2]))

    ug = vtkUnstructuredGrid()
    ug.SetPoints(cell.GetPoints())
    ug.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
    return ug


def make_quadratic_polygon():
    number_of_vertices = 8

    quadratic_polygon = vtkQuadraticPolygon()

    quadratic_polygon.points.SetNumberOfPoints(8)

    quadratic_polygon.points.SetPoint(0, 0.0, 0.0, 0.0)
    quadratic_polygon.points.SetPoint(1, 2.0, 0.0, 0.0)
    quadratic_polygon.points.SetPoint(2, 2.0, 2.0, 0.0)
    quadratic_polygon.points.SetPoint(3, 0.0, 2.0, 0.0)
    quadratic_polygon.points.SetPoint(4, 1.0, 0.0, 0.0)
    quadratic_polygon.points.SetPoint(5, 2.0, 1.0, 0.0)
    quadratic_polygon.points.SetPoint(6, 1.0, 2.0, 0.0)
    quadratic_polygon.points.SetPoint(7, 0.0, 1.0, 0.0)
    quadratic_polygon.points.SetPoint(5, 3.0, 1.0, 0.0)

    quadratic_polygon.point_ids.SetNumberOfIds(number_of_vertices)
    for i in range(0, number_of_vertices):
        quadratic_polygon.point_ids.SetId(i, i)

    ug = vtkUnstructuredGrid(points=quadratic_polygon.points)
    ug.SetPoints(quadratic_polygon.GetPoints())
    ug.InsertNextCell(quadratic_polygon.cell_type, quadratic_polygon.point_ids)

    return ug


def make_quadratic_pyramid():
    cell = vtkQuadraticPyramid()
    pcoords = cell.GetParametricCoords()
    for i in range(0, cell.number_of_points):
        cell.point_ids.SetId(i, i)
        cell.points.SetPoint(i, (pcoords[3 * i]), (pcoords[3 * i + 1]), (pcoords[3 * i + 2]))

    ug = vtkUnstructuredGrid(points=cell.points)
    ug.SetPoints(cell.GetPoints())
    ug.InsertNextCell(cell.cell_type, cell.point_ids)

    t = vtkTransform()
    t.RotateX(-90)
    t.Translate(0, 0, 0)

    tf = vtkTransformFilter()
    tf.SetTransform(t)
    tf.SetInputData(ug)
    tf.Update()

    # Put the transformed points back.
    ug.SetPoints(tf.GetOutput().GetPoints())

    return ug


def make_tile(bounds, expansion_factor=0.5, thickness_ratio=0.05, shift_y=-0.05):
    """
    Make a tile slightly larger or smaller than the bounds in the
      X and Z directions and thinner or thicker in the Y direction.

    A thickness_ratio of zero reduces the tile to an XZ plane.

    :param bounds: The bounds for the tile.
    :param expansion_factor: The expansion factor in the XZ plane.
    :param thickness_ratio: The thickness ratio in the Y direction, >= 0.
    :param shift_y: Used to shift the centre of the plinth in the Y-direction.
    :return: An actor corresponding to the tile.
    """

    d_xyz = (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    )
    thickness = d_xyz[2] * abs(thickness_ratio)
    center = ((bounds[1] + bounds[0]) / 2.0,
              bounds[2] - thickness / 2.0 + shift_y,
              (bounds[5] + bounds[4]) / 2.0)
    x_length = bounds[1] - bounds[0] + (d_xyz[0] * expansion_factor)
    z_length = bounds[5] - bounds[4] + (d_xyz[2] * expansion_factor)
    plane = vtkCubeSource()
    plane.SetCenter(center)
    plane.SetXLength(x_length)
    plane.SetYLength(thickness)
    plane.SetZLength(z_length)

    plane_mapper = vtkPolyDataMapper()
    plane_mapper.SetInputConnection(plane.GetOutputPort())

    tile_actor = vtkActor()
    tile_actor.SetMapper(plane_mapper)

    return tile_actor


def get_text_property():
    colors = vtkNamedColors()

    pty = vtkTextProperty()
    pty.BoldOn()
    pty.SetJustificationToCentered()
    pty.SetColor(colors.GetColor3d('Black'))
    return pty


def get_label_property():
    colors = vtkNamedColors()

    pty = vtkTextProperty()
    pty.BoldOn()
    pty.ShadowOn()
    pty.SetJustificationToCentered()
    pty.SetColor(colors.GetColor3d('DeepPink'))
    return pty


def get_back_face_property():
    colors = vtkNamedColors()

    pty = vtkProperty()
    pty.SetAmbientColor(colors.GetColor3d('LightSalmon'))
    pty.SetDiffuseColor(colors.GetColor3d('OrangeRed'))
    pty.SetSpecularColor(colors.GetColor3d('White'))
    pty.SetSpecular(0.2)
    pty.SetDiffuse(1.0)
    pty.SetAmbient(0.2)
    pty.SetSpecularPower(20.0)
    pty.SetOpacity(1.0)
    return pty


def get_actor_property():
    colors = vtkNamedColors()

    pty = vtkProperty()
    pty.SetAmbientColor(colors.GetColor3d('DarkSalmon'))
    pty.SetDiffuseColor(colors.GetColor3d('Seashell'))
    pty.SetSpecularColor(colors.GetColor3d('White'))
    pty.SetSpecular(0.5)
    pty.SetDiffuse(0.7)
    pty.SetAmbient(0.5)
    pty.SetSpecularPower(20.0)
    pty.SetOpacity(0.9)
    pty.EdgeVisibilityOn()
    pty.SetLineWidth(3)
    return pty


def get_point_actor_property():
    colors = vtkNamedColors()

    pty = vtkProperty()
    pty.SetAmbientColor(colors.GetColor3d('Gold'))
    pty.SetDiffuseColor(colors.GetColor3d('Yellow'))
    pty.SetSpecularColor(colors.GetColor3d('White'))
    pty.SetSpecular(0.5)
    pty.SetDiffuse(0.7)
    pty.SetAmbient(0.5)
    pty.SetSpecularPower(20.0)
    pty.SetOpacity(1.0)
    return pty


def get_tile_property():
    colors = vtkNamedColors()

    pty = vtkProperty()
    pty.SetAmbientColor(colors.GetColor3d('SteelBlue'))
    pty.SetDiffuseColor(colors.GetColor3d('LightSteelBlue'))
    pty.SetSpecularColor(colors.GetColor3d('White'))
    pty.SetSpecular(0.5)
    pty.SetDiffuse(0.7)
    pty.SetAmbient(0.5)
    pty.SetSpecularPower(20.0)
    pty.SetOpacity(0.8)
    pty.EdgeVisibilityOn()
    pty.SetLineWidth(1)
    return pty


def vtk_version_ok(major, minor, build):
    """
    Check the VTK version.

    :param major: Major version.
    :param minor: Minor version.
    :param build: Build version.
    :return: True if the requested VTK version is greater or equal to the actual VTK version.
    """
    needed_version = 10000000000 * int(major) + 100000000 * int(minor) + int(build)
    try:
        vtk_version_number = VTK_VERSION_NUMBER
    except AttributeError:  # as error:
        ver = vtkVersion()
        vtk_version_number = 10000000000 * ver.GetVTKMajorVersion() + 100000000 * ver.GetVTKMinorVersion() \
                             + ver.GetVTKBuildVersion()
    if vtk_version_number >= needed_version:
        return True
    else:
        return False


if __name__ == '__main__':
    main()