import torch
from safetensors.torch import load_file, load_model
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
from transformers.modeling_outputs import ImageSuperResolutionOutput

from .basic import BasicOP
from .richardson import RichardsonLucy
from .gsasr.gsasr import GSASR

__all__ = ["get_model"]

# TODO: Add a diffusion method to train
# TODO: Fix RichardsonLucy and Basic method's contrast for Sum and Mean approach


def __load_model(model: torch.nn.Module, checkpoint_path: str):
    try:
        load_model(model, checkpoint_path)
    except RuntimeError:
        state_dict = load_file(checkpoint_path)
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                clean_key = key.replace("_orig_mod.", "", 1)
            else:
                clean_key = key
            clean_state_dict[clean_key] = value
        model.load_state_dict(clean_state_dict)
        # save_model(model, checkpoint_path)
    except FileNotFoundError:
        print(f"Could not find file {checkpoint_path}. Aborting model loading.")
    return model


def get_model(model_name: str, model_config: str, checkpoint: str):
    if "Swin2SR" == model_name:
        cfg = Swin2SRConfig.from_json_file(model_config)
        model = Swin2SRForImageSuperResolution(cfg)

        if checkpoint is not None:
            model = __load_model(model, f"{checkpoint}/model.safetensors")

        def preprocess_fn(**kwargs):
            return {"pixel_values": kwargs["pixel_values"]}

        def postprocess_fn(output: ImageSuperResolutionOutput):
            return output.reconstruction
    
    elif "GSASR" == model_name:
        model = GSASR.from_json_file(model_config)
        
        if checkpoint is not None:
            model == __load_model(model, f"{checkpoint}/model.safetensors")
        
        def preprocess_fn(**kwargs):
            return {"pixel_values": kwargs["pixel_values"], "upscale": kwargs["upscale"]}

        def postprocess_fn(output: torch.Tensor):
            return output

    elif "RL" == model_name:
        model = RichardsonLucy.from_json_file(model_config)

        def preprocess_fn(**kwargs):
            return {"pixel_values": kwargs["pixel_values"], "psf": kwargs["psf"][0]}

        def postprocess_fn(output: torch.Tensor, target: torch.Tensor = None):
            method = model.op
            if "max" == method:
                processed_img = output
            elif "mean" == method:
                avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                processed_img = output / avg[..., None, None, None]
            elif "sum" == method:
                avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                processed_img = output / avg[..., None, None, None]
            return processed_img

    elif "Basic" == model_name:
        model = BasicOP.from_json_file(model_config)

        def preprocess_fn(**kwargs):
            return {"pixel_values": kwargs["pixel_values"]}

        def postprocess_fn(output: torch.Tensor, target: torch.Tensor = None):
            method = model.op
            if "max" == method:
                processed_img = output
            elif "mean" == method:
                avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                processed_img = output / avg[..., None, None, None]
            elif "sum" == method:
                avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                processed_img = output / avg[..., None, None, None]
            return processed_img

    else:
        raise ValueError(f"Model {model_name} not implemented.")

    return model, preprocess_fn, postprocess_fn
