import torch
import torch.nn.functional as F
from configargparse import Namespace
from safetensors.torch import load_model
from skimage.restoration import richardson_lucy
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
from transformers.modeling_outputs import ImageSuperResolutionOutput


class RichardsonLucy(torch.nn.Module):
    def __init__(
        self,
        upscale: int,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        super().__init__()
        self.upscale = upscale
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def forward(self, pixel_values: torch.Tensor, psf: torch.Tensor):
        device = pixel_values.device
        img = torch.mean(pixel_values, dim=1)[:, None, ...]
        img = F.interpolate(
            img,
            scale_factor=self.upscale,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        B, C, H, W = img.shape
        psf = psf[None, None, ...].expand((B, C, -1, -1))
        output = richardson_lucy(
            img.detach().cpu().numpy(), psf.detach().cpu().numpy()
        )
        output = torch.from_numpy(output).to(device=device)
        return output


class GetMax(torch.nn.Module):
    def __init__(
        self,
        upscale: int,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        super().__init__()
        self.upscale = upscale
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def forward(self, pixel_values: torch.Tensor):
        img = torch.max(pixel_values, dim=1).values[:, None, ...]
        output = F.interpolate(
            img,
            scale_factor=self.upscale,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        return output


def get_model(args: Namespace):
    if "Swin2SR" == args.model_name:
        cfg = Swin2SRConfig(
            image_size=args.second_crop,
            num_channels=args.in_num_channels,
            num_channels_out=args.out_num_channels,
            window_size=args.window_size,
            upscale=args.upscale,
        )
        model = Swin2SRForImageSuperResolution(cfg)

        if args.checkpoint is not None:
            try:
                load_model(model, f"{args.checkpoint}/model.safetensors")
            except FileNotFoundError:
                pass

        def preprocess_fn(**kwargs):
            return {"pixel_values": kwargs["pixel_values"]}

        def postprocess_fn(output: ImageSuperResolutionOutput):
            return output.reconstruction

    elif "RL" == args.model_name:
        model = RichardsonLucy(upscale=args.upscale)

        def preprocess_fn(**kwargs):
            return {"pixel_values": kwargs["pixel_values"], "psf": kwargs["psf"][0]}

        def postprocess_fn(output: torch.Tensor):
            return output

    elif "Max" == args.model_name:
        model = GetMax(upscale=args.upscale)

        def preprocess_fn(**kwargs):
            return {"pixel_values": kwargs["pixel_values"]}

        def postprocess_fn(output: torch.Tensor):
            return output

    else:
        raise ValueError(f"Model {args.model_name} not implemented.")

    return model, preprocess_fn, postprocess_fn


if "__main__" == __name__:
    from .parser import parse_arguments

    args = parse_arguments(is_test=True)

    if 1 == args.test:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model: Swin2SRForImageSuperResolution = get_model(args)
        model = model.to(device=device)
        print("==== Loaded model in single GPU ====")
        print(f"Model size: {sum([p.numel() for p in model.parameters()])}")
        model.eval()

        with torch.no_grad():
            lr_image = torch.ones((1, 25, 256, 256), device=device)
            hr_image = model(lr_image).reconstruction

        print(f"LR shape: {lr_image.shape}")
        print(f"HR shape: {hr_image.shape}")
    if 2 == args.test:
        from accelerate import Accelerator

        accelerator = Accelerator()
        device = accelerator.device

        model = get_model(args)
        model = accelerator.prepare(model)

        print(f"==== Loaded model in {accelerator.num_processes} GPU ====")
        print(
            f"Model size in GPU {device}: {sum([p.numel() for p in model.parameters()])}"
        )
        model.eval()

        with torch.no_grad():
            lr_image = torch.ones((1, 25, 256, 256), device=device)
            hr_image = model(lr_image).reconstruction

        print(f"LR shape in GPU {device}: {lr_image.shape}")
        print(f"HR shape in GPU {device}: {hr_image.shape}")

        accelerator.end_training()
