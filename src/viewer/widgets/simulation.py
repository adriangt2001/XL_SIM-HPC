import time
from typing import Any, Literal

import napari
import numpy as np
import torch
from magicclass.widgets import CollapsibleContainer
from magicgui import magicgui
from magicgui.widgets import Container, PushButton


def generation_pipeline(image):
    print("Generating image...")
    time.sleep(1)
    print("DONE")
    return image


def make_basic_settings():
    @magicgui(
        call_button=False,
        num_images={"min": 1, "max": 1000},
        image_size={"min": 64, "max": 2048},
    )
    def basic_settings(
        num_images: int = 10,
        image_size: int = 512,
    ):
        pass

    basic_settings.label = None
    container = CollapsibleContainer(
        text="Basic Settings", widgets=[basic_settings], labels=False, collapsed=True
    )

    return container, basic_settings


def make_camera_settings():
    @magicgui(
        call_button=False,
        # resolution={"min": 1, "max": 1000},
        # image_size={"min": 64, "max": 2048},
        resolution={"label": "Resolution"},
        focal={"label": "Focal", "min": 0},
        magnification={"label": "Magnification", "min": 0},
        axial_fov={"label": "Axial FOV", "min": 0},
        numerical_aperture={"label": "Numerical Aperture", "min": 0},
    )
    def camera_settings(
        resolution: tuple[int, int] = (512, 512),
        focal: int = 200,
        magnification: float = 60,
        axial_fov: float = 30,
        numerical_aperture: float = 1.2,
    ):
        pass

    camera_settings.label = None
    container = CollapsibleContainer(
        text="Camera Settings", widgets=[camera_settings], labels=False, collapsed=True
    )

    return container, camera_settings


def make_sensor_settings():
    @magicgui(
        call_button=False,
        cam_pix={"label": "Pixel (µm)"},
    )
    def sensor_settings(
        cam_pix: float = 6.5,
    ):
        pass

    sensor_settings.label = None
    container = CollapsibleContainer(
        text="Sensor Settings", widgets=[sensor_settings], labels=False, collapsed=True
    )

    return container, sensor_settings


def make_pattern_settings():
    @magicgui(
        call_button=False,
        pattern_case={"label": "Type"},
        periodicity={"label": "Periodicity"},
        # num_aods={"label": "#AODs"},
        aod_propag={"label": "AOD propagation"},
        phase_optim={"label": "Phase Optim"},
    )
    def pattern_settings(
        pattern_case: Literal["multipoint", "multiline", "const_square"] = "multipoint",
        periodicity: float = 1.0,
        # num_aods: int = 1,
        aod_propag: int = 1,
        phase_optim: int = 1,
    ):
        pass

    pattern_settings.label = None
    container = CollapsibleContainer(
        text="Pattern Settings",
        widgets=[pattern_settings],
        labels=False,
        collapsed=True,
    )

    return container, pattern_settings


def make_noise_settings():
    @magicgui(
        call_button=False,
        noise_level={"min": 0.0, "max": 1.0, "step": 0.01},
    )
    def noise_settings(
        noise_level: float = 0.1,
        normalize: bool = False,
    ):
        pass

    noise_settings.label = None
    container = CollapsibleContainer(
        text="Noise Settings", widgets=[noise_settings], labels=False, collapsed=True
    )

    return container, noise_settings


def make_settings_i_dont_know_where_to_put():
    @magicgui(
        call_button=False,
        binn_simu={"label": "Simulation Downsampling"},
        single_plane={"label": "Single Plane"},
        wavelength_ex={"label": "Excitation Wavelength"},
        wavelength_em={"label": "Emission Wavelength"},
    )
    def settings_i_dont_know_where_to_put(
        binn_simu: int = 2,
        single_plane: bool = True,
        wavelength_ex: float = 488,
        wavelength_em: float = 520,
    ):
        pass

    settings_i_dont_know_where_to_put.label = None
    container = CollapsibleContainer(
        text="Settings I Don't Know Where To Put",
        widgets=[settings_i_dont_know_where_to_put],
        labels=False,
        collapsed=True,
    )

    return container, settings_i_dont_know_where_to_put


def make_output_settings():
    @magicgui(
        call_button=False,
    )
    def output_settings(
        output_format: Literal["png", "tiff", "jpg"] = "png",
    ):
        pass

    output_settings.label = None
    container = CollapsibleContainer(
        text="Output Settings", widgets=[output_settings], labels=False, collapsed=True
    )
    return container, output_settings


def make_widget(viewer: napari.Viewer, state: Any):
    generation_container = Container(labels=False)

    basic_settings_container, basic_settings = make_basic_settings()
    camera_settings_container, camera_settings = make_camera_settings()
    sensor_settings_container, sensor_settings = make_sensor_settings()
    pattern_settings_container, pattern_settings = make_pattern_settings()
    noise_settings_container, noise_settings = make_noise_settings()
    settings_i_dont_know_where_to_put_container, settings_i_dont_know_where_to_put = (
        make_settings_i_dont_know_where_to_put()
    )
    output_settings_container, output_settings = make_output_settings()

    generation_container.extend(
        [
            basic_settings_container,
            camera_settings_container,
            sensor_settings_container,
            pattern_settings_container,
            noise_settings_container,
            settings_i_dont_know_where_to_put_container,
            output_settings_container,
        ]
    )

    # Run button
    def run_generation_clicked():
        selected_layers = list(viewer.layers.selection)

        all_settings = {
            "basic": basic_settings.asdict(),
            "camera": camera_settings.asdict(),
            "sensor": sensor_settings.asdict(),
            "pattern": pattern_settings.asdict(),
            "noise": noise_settings.asdict(),
            "settings_i_dont_know_where_to_put": settings_i_dont_know_where_to_put.asdict(),
            "output": output_settings.asdict(),
        }

        state.update_microscope_if_needed(all_settings)

        for layer in selected_layers:
            print(f"Input image shape: {layer.data.shape}")
            img = torch.from_numpy(layer.data)
            generated, calibs = state.simulation_pipeline(img)
            generated = generated.numpy()
            state.last_calib = calibs.numpy()
            print(f"Output image shape: {generated.shape}")
            viewer.add_image(
                generated, name="(generated)" + layer.name
            )

    run_generation_btn = PushButton(text="Run Simulation Pipeline")
    run_generation_btn.clicked.connect(run_generation_clicked)
    generation_container.append(run_generation_btn)

    return generation_container
