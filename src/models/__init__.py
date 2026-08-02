import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, load_model
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
from transformers.modeling_outputs import ImageSuperResolutionOutput

from .basic import BasicOP
from .burstormer import Burstormer
from .gsasr.gsasr import GSASR
from .hat import HAT
from .richardson import RichardsonLucy
from .xlsim import XLSIM

__all__ = ["get_model"]


def __load_model(model: torch.nn.Module, checkpoint_path: str, lora: bool):
    try:
        load_model(model, checkpoint_path)
    except RuntimeError:
        state_dict = load_file(checkpoint_path)
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                clean_key = (
                    key.replace("_orig_mod.", "base_model.model.", 1)
                    if lora
                    else key.replace("_orig_mod.", "", 1)
                )
            else:
                clean_key = key
            clean_state_dict[clean_key] = value
        model.load_state_dict(clean_state_dict)
        # save_model(model, checkpoint_path)
    except FileNotFoundError:
        print(f"Could not find file {checkpoint_path}. Aborting model loading.")
    return model


def get_model(
    model_name: str,
    model_config: str,
    checkpoint: str,
    load_lora: bool,
    lora: bool = False,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    lora_target_modules: list[str] | None = None,
    lora_bias: str | None = None,
):
    match model_name:
        case "Swin2SR":
            cfg = Swin2SRConfig.from_json_file(model_config)
            model = Swin2SRForImageSuperResolution(cfg)

            def preprocess_fn(**kwargs):
                return {"pixel_values": kwargs["pixel_values"]}

            def postprocess_fn(output: ImageSuperResolutionOutput):
                return output.reconstruction

        case "GSASR":
            model = GSASR.from_json_file(model_config)

            def preprocess_fn(**kwargs):
                return {
                    "pixel_values": kwargs["pixel_values"],
                    "upscale": kwargs["upscale"],
                }

            def postprocess_fn(output: torch.Tensor):
                return output

        case "Burstormer":
            model = Burstormer.from_json_file(model_config)

            def preprocess_fn(**kwargs):
                return {"pixel_values": kwargs["pixel_values"]}

            def postprocess_fn(output: torch.Tensor):
                return output

        case "HAT":
            model = HAT.from_json_file(model_config)

            def preprocess_fn(**kwargs):
                return {"pixel_values": kwargs["pixel_values"]}

            def postprocess_fn(output: torch.Tensor):
                return output

        case "XLSIM":
            model = XLSIM.from_json_file(model_config)

            def preprocess_fn(**kwargs):
                pixel_values: torch.Tensor = kwargs["pixel_values"]
                B, S, H, W = pixel_values.shape
                pixel_values = pixel_values.reshape(B, S, 1, H, W)
                return {"pixel_values": pixel_values}

            def postprocess_fn(output: torch.Tensor):
                return output
            
        case "RL":
            model = RichardsonLucy.from_json_file(model_config)

            def preprocess_fn(**kwargs):
                return {"pixel_values": kwargs["pixel_values"], "psf": kwargs["psf"][0]}

            def postprocess_fn(output: torch.Tensor, target: torch.Tensor = None):
                processed_img = output
                # method = model.op
                # if "max" == method:
                #     processed_img = output
                # elif "mean" == method:
                #     avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                #     processed_img = output / avg[..., None, None, None]
                # elif "sum" == method:
                #     avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                #     processed_img = output / avg[..., None, None, None]
                return processed_img

        case "Basic":
            model = BasicOP.from_json_file(model_config)

            def preprocess_fn(**kwargs):
                return {"pixel_values": kwargs["pixel_values"]}

            def postprocess_fn(output: torch.Tensor, target: torch.Tensor = None):
                processed_img = output
                # method = model.op
                # if "max" == method:
                #     processed_img = output
                # elif "mean" == method:
                #     avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                #     processed_img = output / avg[..., None, None, None]
                # elif "sum" == method:
                #     avg = torch.mean(output / target, dim=list(range(1, output.ndim)))
                #     processed_img = output / avg[..., None, None, None]
                return processed_img

        case _:
            raise ValueError(
                f"{model_name} not implemented. Feel free to add it inside models/__init__.py."
            )

    if not load_lora and checkpoint is not None:
        model = __load_model(model, f"{checkpoint}/model.safetensors", load_lora)

    if lora:
        target_modules = (
            lora_target_modules[0]
            if len(lora_target_modules) == 1 and lora_target_modules[0] == "all-linear"
            else lora_target_modules
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=lora_bias,
        )
        model = get_peft_model(model, peft_config=lora_config)

        if load_lora and checkpoint is not None:
            model = __load_model(model, f"{checkpoint}/model.safetensors", lora)

    return model, preprocess_fn, postprocess_fn
