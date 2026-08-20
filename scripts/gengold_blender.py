# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================
import os
import tempfile
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import pyvale.blender as blender
import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens


def get_sample_scene_2d():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x", "disp_y")
    sim_data = sens.scale_length_units(1000.0, sim_data, disp_comps)
    render_mesh = sens.create_render_mesh(
        sim_data,
        ("disp_y", "disp_x"),
        sim_spat_dim=sens.EDim.TWOD,
        field_disp_keys=disp_comps,
    )

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = sens.CameraData(
        pixels_num=np.array([20, 20]),
        pixels_size=np.array([0.00345, 0.00345]),
        pos_world=(0, 0, 500),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        roi_cent_world=(0, 0, 0),
        focal_length=15,
    )
    camera = scene.add_camera(cam_data)
    light_data = blender.LightData(
        type=blender.LightType.POINT,
        pos_world=(0, 0, 400),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        energy=1,
    )
    light = scene.add_light(light_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    return render_mesh, part, cam_data, scene


def get_sample_scene_2d_no_light():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x", "disp_y")
    sim_data = sens.scale_length_units(1000.0, sim_data, disp_comps)
    render_mesh = sens.create_render_mesh(
        sim_data,
        ("disp_y", "disp_x"),
        sim_spat_dim=sens.EDim.TWOD,
        field_disp_keys=disp_comps,
    )

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = sens.CameraData(
        pixels_num=np.array([20, 20]),
        pixels_size=np.array([0.00345, 0.00345]),
        pos_world=(0, 0, 500),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        roi_cent_world=(0, 0, 0),
        focal_length=15,
    )
    camera = scene.add_camera(cam_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    return cam_data, scene


def get_sample_scene_2d_no_cam():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x", "disp_y")
    sim_data = sens.scale_length_units(1000.0, sim_data, disp_comps)
    render_mesh = sens.create_render_mesh(
        sim_data,
        ("disp_y", "disp_x"),
        sim_spat_dim=2,
        field_disp_keys=disp_comps,
    )

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    light_data = blender.LightData(
        type=blender.LightType.POINT,
        pos_world=(0, 0, 400),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        energy=1,
    )
    light = scene.add_light(light_data)

    return part, scene


def get_sample_scene_3d_no_cam():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x", "disp_y")
    sim_data = sens.scale_length_units(1000.0, sim_data, disp_comps)
    render_mesh = sens.create_render_mesh(
        sim_data,
        ("disp_y", "disp_x"),
        sim_spat_dim=sens.EDim.TWOD,
        field_disp_keys=disp_comps,
    )

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    light_data = blender.LightData(
        type=blender.LightType.POINT,
        pos_world=(0, 0, 400),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        energy=1,
    )
    light = scene.add_light(light_data)
    cam_data_0 = sens.CameraData(
        pixels_num=np.array([20, 20]),
        pixels_size=np.array([0.00345, 0.00345]),
        pos_world=np.array([0, 0, 500]),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        roi_cent_world=(0, 0, 0),
        focal_length=15,
    )
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data_0)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    return cam_data_0, part, render_mesh, scene


def get_sample_stereo_scene():
    cam_data_0, part, render_mesh, scene = get_sample_scene_3d_no_cam()
    stereo_system = sens.CameraTools.faceon_stereo_cameras(
        cam_data_0=cam_data_0, stereo_angle=15.0
    )
    cam0, cam1 = scene.add_stereo_system(stereo_system)
    return (stereo_system, part, render_mesh, scene)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    gold_2d_dir = repo_root / "tests" / "blender" / "2D_gold"
    gold_3d_dir = repo_root / "tests" / "blender" / "3D_gold"

    os.makedirs(gold_2d_dir, exist_ok=True)
    os.makedirs(gold_3d_dir, exist_ok=True)

    print("Generating 2D Blender Gold Outputs...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. half_watt_lighting & three_watt_lighting
        for energy, name in [
            (0.5, "half_watt_lighting"),
            (3.0, "three_watt_lighting"),
        ]:
            cam_data, scene = get_sample_scene_2d_no_light()
            light_data = blender.LightData(
                type=blender.LightType.POINT,
                pos_world=(0, 0, 400),
                rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                energy=energy,
            )
            scene.add_light(light_data)
            render_data = blender.RenderData(
                cam_data=cam_data, base_dir=tmp_path
            )
            image_array = scene.render_single_image(
                stage_image=True, render_data=render_data
            )
            np.save(gold_2d_dir / f"{name}.npy", image_array)

        # 2. vertical_cam & horizontal_cam
        for pixels, name in [
            (np.array([10, 20]), "vertical_cam"),
            (np.array([20, 10]), "horizontal_cam"),
        ]:
            part, scene = get_sample_scene_2d_no_cam()
            cam_data = sens.CameraData(
                pixels_num=pixels,
                pixels_size=np.array([0.00345, 0.00345]),
                pos_world=(0, 0, 500),
                rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                roi_cent_world=(0, 0, 0),
                focal_length=15,
            )
            scene.add_camera(cam_data)
            material_data = blender.MaterialData()
            speckle_path = dataset.dic_pattern_5mpx_path()
            mm_px_res = sens.CameraTools.calculate_mm_px_resolution(cam_data)
            scene.add_speckle(
                part=part,
                speckle_path=speckle_path,
                mat_data=material_data,
                mm_px_resolution=mm_px_res,
            )
            render_data = blender.RenderData(
                cam_data=cam_data, base_dir=tmp_path
            )
            image_array = scene.render_single_image(
                stage_image=True, render_data=render_data
            )
            np.save(gold_2d_dir / f"{name}.npy", image_array)

        # 3. cam_from_resolution
        part, scene = get_sample_scene_2d_no_cam()
        pixels_num = np.array([20, 20])
        pixels_size = np.array([0.00345, 0.00345])
        cam_data = sens.CameraTools.blender_camera_from_resolution(
            pixels_num, pixels_size, 500, 0.1
        )
        scene.add_camera(cam_data)
        material_data = blender.MaterialData()
        speckle_path = dataset.dic_pattern_5mpx_path()
        mm_px_res = sens.CameraTools.calculate_mm_px_resolution(cam_data)
        scene.add_speckle(
            part=part,
            speckle_path=speckle_path,
            mat_data=material_data,
            mm_px_resolution=mm_px_res,
        )
        render_data = blender.RenderData(cam_data=cam_data, base_dir=tmp_path)
        image_array = scene.render_single_image(
            stage_image=True, render_data=render_data
        )
        np.save(gold_2d_dir / "cam_from_resolution.npy", image_array)

        # 4. deformed_images
        render_mesh, part, cam_data, scene = get_sample_scene_2d()
        render_data = blender.RenderData(cam_data=cam_data, base_dir=tmp_path)
        image_arrays = scene.render_deformed_images(
            render_mesh,
            sim_spat_dim=3,
            render_data=render_data,
            part=part,
            stage_image=True,
        )
        np.save(gold_2d_dir / "deformed_images.npy", image_arrays[:, :, 10])

        # 5. samples_four & samples_twelve
        for s_val, name in [(4, "samples_four"), (12, "samples_twelve")]:
            render_mesh, part, cam_data, scene = get_sample_scene_2d()
            render_data = blender.RenderData(
                cam_data=cam_data, base_dir=tmp_path, samples=s_val
            )
            image_array = scene.render_single_image(
                stage_image=True, render_data=render_data
            )
            np.save(gold_2d_dir / f"{name}.npy", image_array)

        # 6. bounces_two & bounces_hundred
        for b_val, name in [(2, "bounces_two"), (100, "bounces_hundred")]:
            render_mesh, part, cam_data, scene = get_sample_scene_2d()
            render_data = blender.RenderData(
                cam_data=cam_data, base_dir=tmp_path, max_bounces=b_val
            )
            image_array = scene.render_single_image(
                stage_image=True, render_data=render_data
            )
            np.save(gold_2d_dir / f"{name}.npy", image_array)

        # 7. cycles_engine & eevee_engine
        for engine, name in [
            (blender.RenderEngine.CYCLES, "cycles_engine"),
            (blender.RenderEngine.EEVEE, "eevee_engine"),
        ]:
            render_mesh, part, cam_data, scene = get_sample_scene_2d()
            render_data = blender.RenderData(
                cam_data=cam_data, base_dir=tmp_path, engine=engine
            )
            image_array = scene.render_single_image(
                stage_image=True, render_data=render_data
            )
            np.save(gold_2d_dir / f"{name}.npy", image_array)

    print("Generating 3D Blender Gold Outputs...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. stereo_symmetric
        cam_data_0, part, render_mesh, scene = get_sample_scene_3d_no_cam()
        stereo_system = sens.CameraTools.symmetric_stereo_cameras(
            cam_data_0=cam_data_0, stereo_angle=15.0
        )
        scene.add_stereo_system(stereo_system)
        render_data = blender.RenderData(
            cam_data=(stereo_system.cam_data_0, stereo_system.cam_data_1),
            base_dir=tmp_path,
        )
        image_array = scene.render_single_image(
            stage_image=True, render_data=render_data
        )
        np.save(gold_3d_dir / "stereo_symmetric.npy", image_array)

        # 2. stereo_faceon
        cam_data_0, part, render_mesh, scene = get_sample_scene_3d_no_cam()
        stereo_system = sens.CameraTools.faceon_stereo_cameras(
            cam_data_0=cam_data_0, stereo_angle=15.0
        )
        scene.add_stereo_system(stereo_system)
        render_data = blender.RenderData(
            cam_data=(stereo_system.cam_data_0, stereo_system.cam_data_1),
            base_dir=tmp_path,
        )
        image_array = scene.render_single_image(
            stage_image=True, render_data=render_data
        )
        np.save(gold_3d_dir / "stereo_faceon.npy", image_array)

        # 3. deformed_images
        stereo_system, part, render_mesh, scene = get_sample_stereo_scene()
        render_data = blender.RenderData(
            cam_data=(stereo_system.cam_data_0, stereo_system.cam_data_1),
            base_dir=tmp_path,
        )
        image_arrays = scene.render_deformed_images(
            render_mesh=render_mesh,
            sim_spat_dim=3,
            render_data=render_data,
            part=part,
            stage_image=True,
        )
        np.save(gold_3d_dir / "deformed_images.npy", image_arrays[:, :, 120:])

        # 4. calib_dict
        calib_dict_data = {
            "Cam0_Fx [pixels]": 4347.826086956522,
            "Cam0_Fy [pixels]": 4347.826086956522,
            "Cam0_Fs [pixels]": 0,
            "Cam0_Kappa 1": 0.0,
            "Cam0_Kappa 2": 0.0,
            "Cam0_Kappa 3": 0.0,
            "Cam0_P1": 0.0,
            "Cam0_P2": 0.0,
            "Cam0_Cx [pixels]": 10.0,
            "Cam0_Cy [pixels]": 10.0,
            "Cam1_Fx [pixels]": 4347.826086956522,
            "Cam1_Fy [pixels]": 4347.826086956522,
            "Cam1_Fs [pixels]": 0,
            "Cam1_Kappa 1": 0.0,
            "Cam1_Kappa 2": 0.0,
            "Cam1_Kappa 3": 0.0,
            "Cam1_P1": 0.0,
            "Cam1_P2": 0.0,
            "Cam1_Cx [pixels]": 10.0,
            "Cam1_Cy [pixels]": 10.0,
            "Tx [mm]": -128.46813489644606,
            "Ty [mm]": 0.0,
            "Tz [mm]": 34.42293299863527,
            "Theta [deg]": 0.0,
            "Phi [deg]": 15.000000000000009,
            "Psi [deg]": 0.0,
        }
        with open(gold_3d_dir / "calib_dict.yaml", "w") as f:
            yaml.safe_dump(calib_dict_data, f)

    print("Gold generation complete.")


if __name__ == "__main__":
    main()
