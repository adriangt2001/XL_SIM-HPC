import inspect
from pathlib import Path

import napari
from magicgui import magicgui
from skimage.data import cells3d
from skimage.io import imread

from src.simulation.microscope import Microscope
from src.simulation.sim_pipeline import SimulatorPipeline
from utils.datasets import get_data

from .widgets import reconstruction, simulation


class State:
    def __init__(self):
        self.simulation_pipeline = SimulatorPipeline()
        self._last_microscope_params = None

    def update_microscope_if_needed(self, settings):
        flat_params = {}
        for category_params in settings.values():
            flat_params.update(category_params)

        sig = inspect.signature(Microscope.__init__)
        micro_params = {k: v for k, v in flat_params.items() if k in sig.parameters}

        if micro_params != self._last_microscope_params:
            print(f"Microscope parameters changed: {set(micro_params.keys())}")
            new_microscope = Microscope(**micro_params)
            self.simulation_pipeline = self.simulation_pipeline.change_microscope(
                new_microscope
            )
            self._last_microscope_params = micro_params.copy()
        else:
            print("Microscope parameters unchanged, reusing instance.")


def main(args):
    # Create dataset and dataloader
    
    global_state = State()

    viewer = napari.Viewer()

    @magicgui(call_button="Load Custom Image", image_path={"mode": "r"})
    def load_custom_image(image_path: Path):
        if image_path and image_path.exists():
            img = imread(image_path)[None, None, ...]
            viewer.add_image(img, name=image_path.name)
            print(f"Loaded image: {image_path.name}")
            print(f"Loaded image shape: {img.shape}")

    @magicgui(auto_call=True)
    def toggle_scale_marker(enable_marker: bool = False):
        raise NotImplementedError("Scale marker not implemented yet.")

    viewer.window.add_dock_widget(load_custom_image, area="right", name="Load Image")
    viewer.window.add_dock_widget(
        simulation.make_widget(viewer, global_state), area="right", name="Simulate"
    )
    viewer.window.add_dock_widget(
        reconstruction.make_widget(viewer, global_state),
        area="right",
        name="Reconstruct",
    )
    viewer.window.add_dock_widget(toggle_scale_marker, area="right", name="Marker")

    napari.run()
