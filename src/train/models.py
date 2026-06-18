from configargparse import Namespace
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
import torch

def get_model(args: Namespace):
    if args.model_name == "Swin2SR":
        if args.checkpoint is not None:
            model = Swin2SRForImageSuperResolution.from_pretrained(args.checkpoint)
        else:
            cfg = Swin2SRConfig(
                image_size=args.second_crop,
                num_channels=args.in_num_channels,
                num_channels_out=args.out_num_channels,
                window_size=args.window_size,
                upscale=args.upscale,
            )
            model = Swin2SRForImageSuperResolution(cfg)
    else:
        raise ValueError(f"Model {args.model_name} not implemented.")

    return model


if "__main__" == __name__:
    from .parser import parse_arguments

    args = parse_arguments(is_test=True)

    if 1 == args.test:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        print(f"Model size in GPU {device}: {sum([p.numel() for p in model.parameters()])}")
        model.eval()

        with torch.no_grad():
            lr_image = torch.ones((1, 25, 256, 256), device=device)
            hr_image = model(lr_image).reconstruction

        print(f"LR shape in GPU {device}: {lr_image.shape}")
        print(f"HR shape in GPU {device}: {hr_image.shape}")

        accelerator.end_training()
