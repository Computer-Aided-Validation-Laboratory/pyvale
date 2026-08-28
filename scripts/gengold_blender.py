# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================
"""Generate committed Blender 2D gold outputs for legacy regression tests."""

import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.render as render
import pyvale.verif.renderverif as renderverif


def _camera(pixels: tuple[int, int] = (20, 20)) -> render.Camera:
    return render.Camera(
        pixels_num=np.asarray(pixels),
        pixels_size=np.array((0.00345, 0.00345)),
        pos_world=np.array((0.0, 0.0, 500.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=15.0,
    )


def _mesh(camera: render.Camera) -> render.Mesh3D:
    sim_data = renderverif.scaled_mechanical_2d()
    resolution = (
        camera.pixels_size[0] * camera.pos_world[2] / camera.focal_length
    )
    shader = render.BlenderTextureShader(
        dataset.dic_pattern_5mpx_path(), resolution
    )
    return render.mesh3d_from_simdata(sim_data, shader, ("disp_x", "disp_y"))


def _render(
    tmp_path: Path,
    camera: render.Camera,
    light_energy: float = 1.0,
    samples: int = 2,
    bounces: int = 12,
    engine: render.EBlenderEngine = render.EBlenderEngine.CYCLES,
) -> np.ndarray:
    renderer = render.Blender(
        render.BlenderConfig(
            tmp_path,
            engine=engine,
            samples=samples,
            max_bounces=bounces,
            threads=1,
        )
    )
    result = renderer.render(
        render.Scene3D(
            (_mesh(camera),),
            (camera,),
            (
                render.Light(
                    render.ELightType.POINT,
                    np.array((0.0, 0.0, 400.0)),
                    np.zeros(3),
                    light_energy,
                ),
            ),
        )
    )
    assert result.images is not None
    return result.images[0, 0, :, :, 0]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gold_2d_dir = repo_root / "tests" / "blender" / "2D_gold"
    gold_2d_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 2D Blender Gold Outputs...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. half_watt_lighting & three_watt_lighting
        for energy, name in (
            (0.5, "half_watt_lighting"),
            (3.0, "three_watt_lighting"),
        ):
            print(f"Generating {name}...")
            img = _render(tmp_path / name, _camera(), light_energy=energy)
            np.save(gold_2d_dir / f"{name}.npy", img)

        # 2. vertical_cam & horizontal_cam
        for pixels, name in (
            ((10, 20), "vertical_cam"),
            ((20, 10), "horizontal_cam"),
        ):
            print(f"Generating {name}...")
            img = _render(tmp_path / name, _camera(pixels))
            np.save(gold_2d_dir / f"{name}.npy", img)

        # 3. samples_four & samples_twelve
        for samples, name in ((4, "samples_four"), (12, "samples_twelve")):
            print(f"Generating {name}...")
            img = _render(tmp_path / name, _camera(), samples=samples)
            np.save(gold_2d_dir / f"{name}.npy", img)

        # 4. bounces_two & bounces_hundred
        for bounces, name in ((2, "bounces_two"), (100, "bounces_hundred")):
            print(f"Generating {name}...")
            img = _render(tmp_path / name, _camera(), bounces=bounces)
            np.save(gold_2d_dir / f"{name}.npy", img)

        # 5. cycles_engine & eevee_engine
        for engine, name in (
            (render.EBlenderEngine.CYCLES, "cycles_engine"),
            (render.EBlenderEngine.EEVEE, "eevee_engine"),
        ):
            print(f"Generating {name}...")
            img = _render(tmp_path / name, _camera(), engine=engine)
            np.save(gold_2d_dir / f"{name}.npy", img)

        # 6. cam_from_resolution
        print("Generating cam_from_resolution...")
        cam_res = render.blender_camera_from_resolution(
            np.array((20, 20)),
            np.array((0.00345, 0.00345)),
            500.0,
            0.1,
        )
        img_res = _render(tmp_path / "cam_res", cam_res)
        np.save(gold_2d_dir / "cam_from_resolution.npy", img_res)

        # 7. deformed_images
        print("Generating deformed_images...")
        camera = _camera()
        renderer = render.Blender(
            render.BlenderConfig(
                tmp_path / "deformed",
                threads=1,
                render_deformed=True,
            )
        )
        result = renderer.render(
            render.Scene3D(
                (_mesh(camera),),
                (camera,),
                (
                    render.Light(
                        render.ELightType.POINT,
                        np.array((0.0, 0.0, 400.0)),
                        np.zeros(3),
                        1.0,
                    ),
                ),
            )
        )
        assert result.images is not None
        np.save(
            gold_2d_dir / "deformed_images.npy",
            result.images[10, 0, :, :, 0],
        )

    print("Finished generating 2D Blender Gold Outputs.")


if __name__ == "__main__":
    main()
