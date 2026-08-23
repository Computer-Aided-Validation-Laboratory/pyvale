# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Finite-element-driven orthographic image deformation.

The module provides the initial :class:`IImageWarp2D` implementation. It
interpolates planar nodal displacements onto an orthographic image grid and
warps a reference greyscale image for each simulation frame.
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage

from pyvale.render.camera import Camera2D
from pyvale.render.imagewarp2d import IImageWarp2D
from pyvale.render.mesh import Mesh2D
from pyvale.render.result import ImageWarpResult
from pyvale.render.cameratools import CameraTools
from pyvale.render.imagetools import EImageType, ImageTools
from pyvale.render.scene import Scene2D
from pyvale.render.verifyinput import mesh_convention_issues


@dataclass(slots=True)
class ImageDefOpts:
    """Options controlling finite-element image deformation.

    Parameters
    ----------
    save_path : pathlib.Path or None, optional
        Directory used by :meth:`ImageDef2D.deform_images_to_disk`. ``None``
        selects a ``deformed_images`` directory in the current directory.
    save_tag : str, optional
        Prefix used for files written to disk.
    mask_input_image : bool, optional
        Mask reference pixels outside the finite-element surface.
    crop_on : bool, optional
        Reserved option controlling input-image cropping.
    crop_px : numpy.ndarray or None, optional
        Reserved crop bounds in pixel coordinates.
    calc_res_from_fe : bool, optional
        Reserved option to calculate output resolution from the FE mesh.
    calc_res_border_px : int, optional
        Reserved pixel border for FE-derived resolution.
    add_static_ref : bool, optional
        Prepend an undeformed reference frame to displacement data.
    fe_interp : str, optional
        Interpolation method passed to :func:`scipy.interpolate.griddata`.
    fe_rescale : bool, optional
        Rescale coordinates before finite-element interpolation.
    fe_extrap_outside_fov : bool, optional
        Move values outside the specimen beyond the field of view.
    image_def_order : int, optional
        Spline order used to interpolate greyscale image values.
    image_def_extrap : str, optional
        Boundary mode passed to :func:`scipy.ndimage.map_coordinates`.
    image_def_extval : float, optional
        Constant value used by the ``constant`` interpolation boundary mode.
    def_complex_geom : bool, optional
        Deform the specimen mask as well as the greyscale image.
    """
    save_path: Path | None = None
    save_tag: str = "defimage"

    mask_input_image: bool = True

    crop_on: bool = False
    crop_px: np.ndarray | None = None

    calc_res_from_fe: bool = False
    calc_res_border_px: int = 5

    add_static_ref: bool = False

    fe_interp: str = "linear"
    fe_rescale: bool = True
    fe_extrap_outside_fov: bool = True

    image_def_order: int = 3
    image_def_extrap: str = "nearest"
    image_def_extval: float = 0.0

    def_complex_geom: bool = True

    def __post_init__(self) -> None:
        """Set the default deformation-image output directory."""
        if self.save_path is None:
            self.save_path = Path.cwd() / "deformed_images"


class ImageDef2D(IImageWarp2D):
    """Warp a planar reference image from finite-element displacements.

    Parameters
    ----------
    options : ImageDefOpts or None, optional
        Deformation options. ``None`` creates :class:`ImageDefOpts` with its
        default values.
    """

    def __init__(self, options: ImageDefOpts | None = None) -> None:
        """Create a planar finite-element image-warp renderer.

        Parameters
        ----------
        options : ImageDefOpts or None, optional
            Deformation options. ``None`` uses default options.
        """
        self.options = ImageDefOpts() if options is None else options

    def verify_input(self, scene: Scene2D) -> None:
        """Verify a planar image-warp request before preprocessing it.

        Parameters
        ----------
        scene : Scene2D
            Complete planar rendering request containing mesh, camera, and
            source image.

        Raises
        ------
        ValueError
            If source image, mesh, or displacement data is inconsistent.
        """
        if scene.source_image is None:
            raise ValueError("Scene2D.source_image is required for ImageDef2D.")

        source_image = scene.source_image
        camera = scene.camera
        mesh = scene.mesh

        if source_image.ndim != 2:
            raise ValueError("source_image must be a two-dimensional image.")

        if mesh.coords.ndim != 2 or mesh.coords.shape[1] < 2:
            raise ValueError("mesh.coords must have at least two coordinate columns.")

        if mesh.connectivity.ndim != 2 or mesh.connectivity.size == 0:
            raise ValueError("mesh.connectivity must be a non-empty rank-2 array.")

        if np.any(mesh.connectivity < 0) or np.any(mesh.connectivity >= mesh.coords.shape[0]):
            raise ValueError("mesh.connectivity contains invalid node indices.")

        convention_issues = mesh_convention_issues(
            mesh.coords,
            mesh.connectivity,
            "mesh",
        )
        if convention_issues:
            raise ValueError(convention_issues[0].message)

        if mesh.displacement.ndim != 3 or mesh.displacement.shape[1:] != (mesh.coords.shape[0], 2):
            raise ValueError("mesh.displacement must have shape (frames, nodes, 2).")

        if source_image.shape != tuple(camera.pixels_count[::-1]):
            raise ValueError("source_image shape must match camera pixels_count.")

    def _render(self, scene: Scene2D) -> ImageWarpResult:
        """Warp every displacement frame in a validated request.

        Parameters
        ----------
        scene : Scene2D
            Validated planar rendering request.

        Returns
        -------
        ImageWarpResult
            Deformed images with frame and singleton-camera axes.
        """
        source_image = scene.source_image
        camera = scene.camera
        mesh = scene.mesh

        assert source_image is not None

        upsampled, mask, _, disp_x, disp_y = self.preprocess(
            camera, source_image.copy(), mesh.coords, mesh.connectivity,
            mesh.displacement[:, :, 0].T, mesh.displacement[:, :, 1].T, self.options,
        )

        assert upsampled is not None and disp_x is not None and disp_y is not None

        images = []
        for frame_index in range(mesh.displacement.shape[0]):
            deformed, _, _, _, _ = self.deform_one_image(
                upsampled, camera, self.options, mesh.coords,
                np.column_stack((disp_x[:, frame_index], disp_y[:, frame_index])),
                mask, print_on=False,
            )
            images.append(deformed)

        return ImageWarpResult(
            images=np.asarray(images)[:, None, :, :, None],
            masks=None,
            output_paths=(),
        )

    @staticmethod
    def image_mask_from_sim(cam_data: Camera2D,
                            image: np.ndarray,
                            coords: np.ndarray,
                            connectivity: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Mask pixels outside a finite-element surface.

        Parameters
        ----------
        cam_data : Camera2D
            Orthographic camera defining the image extent.
        image : numpy.ndarray
            Reference image to mask in place.
        coords : numpy.ndarray
            Finite-element node coordinates.
        connectivity : numpy.ndarray
            Surface-element connectivity table.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Masked image and sub-pixel specimen mask.
        """

        subsample: int = 1

        coords_raster = coords - cam_data.roi_cent_world[:coords.shape[1]]
        if coords_raster.shape[1] >= 3:
            coords_raster = coords_raster[:, :-1]

        coords_raster[:, 0] = 2 * coords_raster[:, 0] / cam_data.field_of_view[0]
        coords_raster[:, 1] = 2 * coords_raster[:, 1] / cam_data.field_of_view[1]

        coords_raster[:, 0] = (coords_raster[:, 0] + 1) / 2 * cam_data.pixels_count[0]
        coords_raster[:, 1] = (1 - coords_raster[:, 1]) / 2 * cam_data.pixels_count[1]

        elem_coords = np.ascontiguousarray(coords_raster[connectivity, :])

        elem_coord_min = np.min(elem_coords, axis=1)
        elem_coord_max = np.max(elem_coords, axis=1)

        crop_mask = np.zeros([elem_coords.shape[0], 4], dtype=np.int8)
        crop_mask[elem_coord_min[:, 0] <= (cam_data.pixels_count[0] - 1), 0] = 1
        crop_mask[elem_coord_min[:, 1] <= (cam_data.pixels_count[1] - 1), 1] = 1
        crop_mask[elem_coord_max[:, 0] >= 0, 2] = 1
        crop_mask[elem_coord_max[:, 1] >= 0, 3] = 1
        crop_mask = np.sum(crop_mask, axis=1) == 4

        elem_coords = np.ascontiguousarray(elem_coords[crop_mask, :, :])

        elem_coord_min = elem_coord_min[crop_mask, :]
        elem_coord_max = elem_coord_max[crop_mask, :]
        num_elems_in_image = elem_coord_min.shape[0]

        elem_bound_boxes_inds = np.zeros([num_elems_in_image, 4], dtype=np.int32)
        elem_bound_boxes_inds[:, 0] = ImageTools.elem_bound_box_low(
            elem_coord_min[:, 0])
        elem_bound_boxes_inds[:, 1] = ImageTools.elem_bound_box_high(
            elem_coord_max[:, 0],
            cam_data.pixels_count[0] - 1)
        elem_bound_boxes_inds[:, 2] = ImageTools.elem_bound_box_low(
            elem_coord_min[:, 1])
        elem_bound_boxes_inds[:, 3] = ImageTools.elem_bound_box_high(
            elem_coord_max[:, 1],
            cam_data.pixels_count[1] - 1)

        num_edges: int = 3
        if elem_coords.shape[1] > 3:
            num_edges = 4

        mask_subpixel_buffer = np.full(subsample * cam_data.pixels_count, 0.0).T
        for ee in range(elem_coords.shape[0]):
            bound_subpx_x = np.arange(elem_bound_boxes_inds[ee, 0],
                                      elem_bound_boxes_inds[ee, 1],
                                      1 / subsample) + 1 / (2 * subsample)
            bound_subpx_y = np.arange(elem_bound_boxes_inds[ee, 2],
                                      elem_bound_boxes_inds[ee, 3],
                                      1 / subsample) + 1 / (2 * subsample)
            (bound_subpx_grid_x, bound_subpx_grid_y) = np.meshgrid(bound_subpx_x,
                                                                   bound_subpx_y)
            bound_coords_grid_shape = bound_subpx_grid_x.shape
            bound_subpx_coords_flat = np.vstack((bound_subpx_grid_x.flatten(),
                                                 bound_subpx_grid_y.flatten()))

            subpx_inds_x = np.arange(subsample * elem_bound_boxes_inds[ee, 0],
                                     subsample * elem_bound_boxes_inds[ee, 1])
            subpx_inds_y = np.arange(subsample * elem_bound_boxes_inds[ee, 2],
                                     subsample * elem_bound_boxes_inds[ee, 3])
            (subpx_inds_grid_x, subpx_inds_grid_y) = np.meshgrid(subpx_inds_x,
                                                                 subpx_inds_y)

            edge = np.zeros((num_edges, bound_subpx_coords_flat.shape[1]), dtype=np.float64)

            if num_edges == 4:
                edge[0, :] = ImageTools.edge_function(elem_coords[ee, 1, :],
                                           elem_coords[ee, 2, :],
                                           bound_subpx_coords_flat)
                edge[1, :] = ImageTools.edge_function(elem_coords[ee, 2, :],
                                           elem_coords[ee, 3, :],
                                           bound_subpx_coords_flat)
                edge[2, :] = ImageTools.edge_function(elem_coords[ee, 3, :],
                                           elem_coords[ee, 0, :],
                                           bound_subpx_coords_flat)
                edge[3, :] = ImageTools.edge_function(elem_coords[ee, 0, :],
                                           elem_coords[ee, 1, :],
                                           bound_subpx_coords_flat)
            else:
                edge[0, :] = ImageTools.edge_function(elem_coords[ee, 1, :],
                                           elem_coords[ee, 2, :],
                                           bound_subpx_coords_flat)
                edge[1, :] = ImageTools.edge_function(elem_coords[ee, 2, :],
                                           elem_coords[ee, 0, :],
                                           bound_subpx_coords_flat)
                edge[2, :] = ImageTools.edge_function(elem_coords[ee, 0, :],
                                           elem_coords[ee, 1, :],
                                           bound_subpx_coords_flat)

            edge_check = np.zeros_like(edge, dtype=np.int8)
            edge_check[edge >= 0.0] = 1
            edge_check = np.sum(edge_check, axis=0)
            edge_mask_flat = edge_check == num_edges
            edge_mask_grid = np.reshape(edge_mask_flat, bound_coords_grid_shape)

            subpx_inds_grid_x = subpx_inds_grid_x[edge_mask_grid]
            subpx_inds_grid_y = subpx_inds_grid_y[edge_mask_grid]
            mask_subpixel_buffer[subpx_inds_grid_y, subpx_inds_grid_x] += 1.0

        mask_subpixel_buffer[mask_subpixel_buffer > 1.0] = 1.0

        mask_buffer = CameraTools.average_subpixel_image(mask_subpixel_buffer,
                                                         subsample)
        image[mask_buffer < 1.0] = cam_data.background
        return (image, mask_subpixel_buffer)


    @staticmethod
    def upsample_image(cam_data: Camera2D,
                       input_im: np.ndarray):
        """Interpolate a reference image onto the camera sub-pixel grid.

        Parameters
        ----------
        cam_data : Camera2D
            Orthographic camera defining pixel and sub-pixel spacing.
        input_im : numpy.ndarray
            Two-dimensional source image.

        Returns
        -------
        numpy.ndarray
            Smoothly interpolated sub-pixel image.
        """
        (px_vec_xm, px_vec_ym) = CameraTools.pixel_vec_leng(cam_data.field_of_view,
                                                            cam_data.leng_per_px)

        (subpx_vec_xm, subpx_vec_ym) = CameraTools.subpixel_vec_leng(
                                                        cam_data.field_of_view,
                                                        cam_data.leng_per_px,
                                                        cam_data.subsample)

        spline_interp = RectBivariateSpline(px_vec_xm,
                                            px_vec_ym,
                                            input_im.T)
        upsampled_image_interp = lambda x_new, y_new: spline_interp(x_new, y_new).T

        upsampled_image = upsampled_image_interp(subpx_vec_xm, subpx_vec_ym)

        return upsampled_image


    @staticmethod
    def preprocess(cam_data: Camera2D,
                   image_input: np.ndarray,
                   coords: np.ndarray,
                   connectivity: np.ndarray,
                   disp_x: np.ndarray,
                   disp_y: np.ndarray,
                   id_opts: ImageDefOpts,
                   print_on: bool = False
                   ) -> tuple[np.ndarray | None,
                              np.ndarray | None,
                              np.ndarray | None,
                              np.ndarray | None,
                              np.ndarray | None]:
        """Prepare image, mask, and displacement data for deformation.

        Parameters
        ----------
        cam_data : Camera2D
            Orthographic camera for the output image.
        image_input : numpy.ndarray
            Reference image to crop, mask, and upsample.
        coords : numpy.ndarray
            Finite-element node coordinates.
        connectivity : numpy.ndarray
            Surface-element connectivity table.
        disp_x, disp_y : numpy.ndarray
            Nodal x and y displacement fields, arranged by node and frame.
        id_opts : ImageDefOpts
            Deformation controls.
        print_on : bool, optional
            Print timing diagnostics while preprocessing.

        Returns
        -------
        tuple[numpy.ndarray or None, numpy.ndarray or None,
        numpy.ndarray or None,
        numpy.ndarray or None, numpy.ndarray or None]
            Upsampled image, sub-pixel mask, prepared input image, and the x
            and y displacement fields.
        """

        if print_on:
            print("\n" + "=" * 80)
            print("IMAGE DEF PRE-PROCESSING\n")

        if not id_opts.save_path.is_dir():
            id_opts.save_path.mkdir()

        if disp_x.ndim == 1:
            disp_x = np.atleast_2d(disp_x).T
        if disp_y.ndim == 1:
            disp_y = np.atleast_2d(disp_y).T

        if id_opts.add_static_ref:
            num_nodes = coords.shape[0]
            disp_x = np.hstack((np.zeros((num_nodes, 1)), disp_x))
            disp_y = np.hstack((np.zeros((num_nodes, 1)), disp_y))

        image_input = CameraTools.crop_image_rectangle(image_input,
                                                       cam_data.pixels_count)

        if id_opts.mask_input_image or id_opts.def_complex_geom:
            if print_on:
                print('Image masking or complex geometry on, getting image mask.')
                tic = time.perf_counter()

            (image_input,
             image_mask) = ImageDef2D.image_mask_from_sim(cam_data,
                                                          image_input,
                                                          coords,
                                                          connectivity)

            if print_on:
                toc = time.perf_counter()
                print(f'Calculating image mask took {toc - tic:.4f} seconds')
        else:
            image_mask = None

        if print_on:
            print('\n' + '-' * 80)
            print('GENERATE UPSAMPLED IMAGE\n')
            print(f'Upsampling input image with a {cam_data.subsample}x{cam_data.subsample} subpixel')
            tic = time.perf_counter()

        upsampled_image = ImageDef2D.upsample_image(cam_data, image_input)

        if print_on:
            toc = time.perf_counter()
            print(f'Upsampling image withtook {toc - tic:.4f} seconds')

        return (upsampled_image, image_mask, image_input, disp_x, disp_y)

    @staticmethod
    def deform_one_image(upsampled_image: np.ndarray,
                         cam_data: Camera2D,
                         id_opts: ImageDefOpts,
                         coords: np.ndarray,
                         disp: np.ndarray,
                         image_mask: np.ndarray | None = None,
                         print_on: bool = True
                         ) -> tuple[np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray | None]:
        """Deform one reference image for one nodal displacement frame.

        Parameters
        ----------
        upsampled_image : numpy.ndarray
            Sub-pixel reference image returned by :meth:`upsample_image`.
        cam_data : Camera2D
            Orthographic output camera.
        id_opts : ImageDefOpts
            Deformation controls.
        coords : numpy.ndarray
            Finite-element node coordinates.
        disp : numpy.ndarray
            One frame of nodal planar displacements with shape ``(nodes, 2)``.
        image_mask : numpy.ndarray or None, optional
            Sub-pixel mask describing the specimen region.
        print_on : bool, optional
            Print timing diagnostics while deforming.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray or None]
            Deformed image, deformed sub-pixel image, x and y sub-pixel
            displacements, and the optional deformed mask.
        """

        if image_mask is not None:
            if (image_mask.shape[0] != cam_data.pixels_count[1]) or (image_mask.shape[1] != cam_data.pixels_count[0]):
                if image_mask.size == 0:
                    warnings.warn('Image mask not specified, using default mask of whole image.')
                else:
                    warnings.warn('Image mask size does not match camera, using default mask of whole image.')
                image_mask = np.ones([cam_data.pixels_count[1], cam_data.pixels_count[0]])

        (px_grid_xm,
         px_grid_ym) = CameraTools.pixel_grid_leng(cam_data.field_of_view,
                                                   cam_data.leng_per_px)

        (subpx_grid_xm,
         subpx_grid_ym) = CameraTools.subpixel_grid_leng(cam_data.field_of_view,
                                                         cam_data.leng_per_px,
                                                         cam_data.subsample)

        if print_on:
            print('Interpolating displacement onto sub-pixel grid.')
            tic = time.perf_counter()

        (subpx_disp_x, subpx_disp_y) = _interp_sim_disp_to_subpx_grid(
                                                                    coords,
                                                                    disp,
                                                                    cam_data,
                                                                    id_opts,
                                                                    subpx_grid_xm,
                                                                    subpx_grid_ym)

        if print_on:
            toc = time.perf_counter()
            print('Interpolating displacement with NaN extrap took {:.4f} seconds'.format(toc - tic))

        if print_on:
            print('Deforming sub-pixel image.')
            tic = time.perf_counter()

        def_image_subpx = _interp_subpx_image(upsampled_image,
                                              subpx_grid_xm - subpx_disp_x,
                                              subpx_grid_ym - subpx_disp_y,
                                              cam_data,
                                              id_opts)

        if print_on:
            toc = time.perf_counter()
            print('Deforming sub-pixel image with ndimage took {:.4f} seconds'.format(toc - tic))

        if print_on:
            tic = time.perf_counter()

        def_image = CameraTools.average_subpixel_image(def_image_subpx,
                                                       cam_data.subsample)

        if print_on:
            toc = time.perf_counter()
            print('Averaging sub-pixel imagetook {:.4f} seconds'.format(toc - tic))

        if id_opts.def_complex_geom:
            if print_on:
                print('Deforming image mask.')
                tic = time.perf_counter()

            (def_image, def_mask) = _deform_image_mask(def_image,
                                                       image_mask,
                                                       px_grid_xm,
                                                       px_grid_ym,
                                                       subpx_disp_x,
                                                       subpx_disp_y,
                                                       cam_data)

            if print_on:
                toc = time.perf_counter()
                print('Deforming image mask with ndimage took {:.4f} seconds'.format(toc - tic))

        else:
            def_mask = None

        def_image = def_image[::-1, :]

        return (def_image, def_image_subpx, subpx_disp_x, subpx_disp_y, def_mask)

    def deform_images_to_disk(self, scene: Scene2D) -> tuple[Path, ...]:
        """Deform every frame and save it as a TIFF image.

        Parameters
        ----------
        scene : Scene2D
            Validated planar rendering request.

        Returns
        -------
        tuple[pathlib.Path, ...]
            Paths to the saved TIFF images.
        """
        if scene.source_image is None:
            raise ValueError("Scene2D.source_image is required for ImageDef2D.")

        source_image = scene.source_image
        camera = scene.camera
        mesh = scene.mesh

        upsampled, mask, _, disp_x, disp_y = self.preprocess(
            camera, source_image.copy(), mesh.coords, mesh.connectivity,
            mesh.displacement[:, :, 0].T, mesh.displacement[:, :, 1].T, self.options,
        )

        assert upsampled is not None and disp_x is not None and disp_y is not None

        output_paths = []
        num_frames = disp_x.shape[1]

        if print_on:
            print('\n' + '=' * 80)
            print('DEFORMING IMAGES')

        ticl = time.perf_counter()

        for ff in range(num_frames):
            if print_on:
                ticf = time.perf_counter()
                print(f'\nDEFORMING FRAME: {ff}')

            disp = np.array((disp_x[:, ff], disp_y[:, ff])).T
            (def_image,
             _,
             _,
             _,
             _) = ImageDef2D.deform_one_image(upsampled,
                                             camera,
                                             self.options,
                                             mesh.coords,
                                             disp,
                                             mask,
                                             print_on=print_on)

            save_file = self.options.save_path / str(f'{self.options.save_tag}_' +
                    f'{ImageTools.get_num_str(im_num=ff, width=4)}' +
                    '.tiff')
            ImageTools.save_image(save_file,
                                  def_image,
                                  EImageType.TIFF,
                                  camera.bits)

            output_paths.append(save_file)

            if print_on:
                tocf = time.perf_counter()
                print(f'DEFORMING FRAME: {ff} took {tocf - ticf:.4f} seconds')

        if print_on:
            tocl = time.perf_counter()
            print('\n' + '-' * 50)
            print(f'Deforming all images took {tocl - ticl:.4f} seconds')
            print('-' * 50)

            print('\n' + '=' * 80)
            print('COMPLETE\n')

        return tuple(output_paths)


def _interp_sim_disp_to_subpx_grid(coords: np.ndarray,
                                   disp: np.ndarray,
                                   cam_data: Camera2D,
                                   id_opts: ImageDefOpts,
                                   subpx_grid_xm: np.ndarray,
                                   subpx_grid_ym: np.ndarray
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate nodal displacements onto the sub-pixel camera grid.

    Parameters
    ----------
    coords : numpy.ndarray
        Finite-element node coordinates.
    disp : numpy.ndarray
        Nodal planar displacements with shape ``(nodes, 2)``.
    cam_data : Camera2D
        Orthographic output camera.
    id_opts : ImageDefOpts
        Finite-element interpolation controls.
    subpx_grid_xm, subpx_grid_ym : numpy.ndarray
        Horizontal and vertical sub-pixel grids in camera coordinates.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Interpolated x and y displacement fields on the sub-pixel grid.
    """

    subpx_disp_x = griddata((coords[:, 0] + disp[:, 0] + cam_data.world_to_cam[0],
                             coords[:, 1] + disp[:, 1] + cam_data.world_to_cam[1]),
                            disp[:, 0],
                            (subpx_grid_xm, subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    subpx_disp_y = griddata((coords[:, 0] + disp[:, 0] + cam_data.world_to_cam[0],
                             coords[:, 1] + disp[:, 1] + cam_data.world_to_cam[1]),
                            disp[:, 1],
                            (subpx_grid_xm, subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    if id_opts.fe_extrap_outside_fov:
        subpx_disp_ext_vals = 2 * cam_data.field_of_view
    else:
        subpx_disp_ext_vals = (0.0, 0.0)

    subpx_disp_x[np.isnan(subpx_disp_x)] = subpx_disp_ext_vals[0]
    subpx_disp_y[np.isnan(subpx_disp_y)] = subpx_disp_ext_vals[1]

    return (subpx_disp_x, subpx_disp_y)


def _interp_subpx_image(upsampled_image: np.ndarray,
                        def_subpx_x: np.ndarray,
                        def_subpx_y: np.ndarray,
                        cam_data: Camera2D,
                        id_opts: ImageDefOpts,
                        ) -> np.ndarray:
    """Sample a sub-pixel image at deformed camera coordinates.

    Parameters
    ----------
    upsampled_image : numpy.ndarray
        Reference image on the sub-pixel grid.
    def_subpx_x, def_subpx_y : numpy.ndarray
        Deformed sub-pixel coordinates in physical camera units.
    cam_data : Camera2D
        Orthographic output camera.
    id_opts : ImageDefOpts
        Image interpolation controls.

    Returns
    -------
    numpy.ndarray
        Deformed sub-pixel image.
    """

    def_subpx_x = def_subpx_x[::-1, :]
    def_subpx_y = def_subpx_y[::-1, :]

    def_subpx_x_in_px = def_subpx_x * (cam_data.subsample / cam_data.leng_per_px) - 0.5
    def_subpx_y_in_px = def_subpx_y * (cam_data.subsample / cam_data.leng_per_px) - 0.5

    def_image_subpx = ndimage.map_coordinates(upsampled_image,
                                            [[def_subpx_y_in_px],
                                            [def_subpx_x_in_px]],
                                            prefilter=True,
                                            order=id_opts.image_def_order,
                                            mode=id_opts.image_def_extrap,
                                            cval=id_opts.image_def_extval)

    def_image_subpx = def_image_subpx[0, :, :].squeeze()

    return def_image_subpx


def _deform_image_mask(def_image: np.ndarray,
                       image_mask: np.ndarray,
                       px_grid_xm: np.ndarray,
                       px_grid_ym: np.ndarray,
                       subpx_disp_x: np.ndarray,
                       subpx_disp_y: np.ndarray,
                       cam_data: Camera2D,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Warp a specimen mask and apply it to a deformed image.

    Parameters
    ----------
    def_image : numpy.ndarray
        Deformed output image to mask in place.
    image_mask : numpy.ndarray
        Reference specimen mask.
    px_grid_xm, px_grid_ym : numpy.ndarray
        Horizontal and vertical output-pixel grids in camera coordinates.
    subpx_disp_x, subpx_disp_y : numpy.ndarray
        Interpolated displacement fields on the sub-pixel grid.
    cam_data : Camera2D
        Orthographic output camera.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Masked deformed image and sampled mask.
    """

    px_disp_x = CameraTools.average_subpixel_image(subpx_disp_x,
                                                   cam_data.subsample)
    px_disp_y = CameraTools.average_subpixel_image(subpx_disp_y,
                                                   cam_data.subsample)

    def_px_x = px_grid_xm - px_disp_x
    def_px_y = px_grid_ym - px_disp_y
    def_px_x = def_px_x[::-1, :]
    def_px_y = def_px_y[::-1, :]

    def_px_x_in_px = def_px_x * (1 / cam_data.leng_per_px) - 0.5
    def_px_y_in_px = def_px_y * (1 / cam_data.leng_per_px) - 0.5

    def_mask = ndimage.map_coordinates(image_mask,
                                        [[def_px_y_in_px],
                                        [def_px_x_in_px]],
                                        prefilter=True,
                                        order=2,
                                        mode='constant',
                                        cval=0)

    def_mask = def_mask[0, :, :].squeeze()
    def_image[def_mask < 0.51] = cam_data.background

    return (def_image, image_mask)


__all__ = ["ImageDef2D", "ImageDefOpts"]