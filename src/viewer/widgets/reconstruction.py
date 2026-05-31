import napari
import torch
from magicclass.widgets import CollapsibleContainer
from magicgui import magicgui
from magicgui.widgets import Container, PushButton

from src.sres.methods import ALGORITHMS


def get_reconstruction_algorithm(name: str):
    return ALGORITHMS[name]


def make_algorithm_settings():
    @magicgui(
        call_button=False,
        # algorithm={"label": "Algorithm"},
        algorithm={"choices": list(ALGORITHMS.keys())}
    )
    def algorithm_settings(
        algorithm = list(ALGORITHMS.keys())[0]
    ):
        pass
    
    algorithm_settings.label = None
    
    container = CollapsibleContainer(text="Algorithm Settings", widgets=[algorithm_settings], labels=False, collapsed=True)

    return container, algorithm_settings

def make_widget(viewer: napari.Viewer, state: 'view.State'):
    widget_container = Container(labels=False)
    
    algorithm_settings_container, algorithm_settings = make_algorithm_settings()
    
    widget_container.extend([
        algorithm_settings_container,
    ])
    
    def run_reconstruction():
        settings = {
            "algorithm": algorithm_settings.asdict()
        }
        
        algorithm = get_reconstruction_algorithm(settings['algorithm']['algorithm'])
        
        
        selected_layers = list(viewer.layers.selection)
        for layer in selected_layers:
            print(f"Input image shape: {layer.data.shape}")
            img = torch.from_numpy(layer.data).to(device=state.simulation_pipeline.device)
            generated = algorithm(img, state)
            generated = generated.numpy()
            print(f"Output image shape: {generated.shape}")
            viewer.add_image(generated, name='(reconstructed)' + layer.name)
        
    reconstruction_button = PushButton(text="Run Reconstruction Pipeline")
    reconstruction_button.clicked.connect(run_reconstruction)
    widget_container.append(reconstruction_button)
    
    return widget_container
    