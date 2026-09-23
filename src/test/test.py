from pathlib import Path

import torch
import torch.nn.functional as F
from configargparse import Namespace
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from data.preprocessing import crop_tensor
from src.data.datasets import get_data
from src.methods import get_model
from src.microscope.microscope import Microscope


def main(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Main model
    model, preprocess_fn, postprocess_fn = get_model(
        args.main_model_name,
        args.main_model_config,
        args.checkpoint,
        args.lora,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
    )
    model = model.to(device=device)
    if args.lora:
        model = model.merge_and_unload()
    model.eval()

    # Comparison
    comparison_models = []
    for model_name, model_config, model_checkpoint in zip(
        args.comparison_model_names,
        args.comparison_model_configs,
        args.comparison_checkpoints,
    ):
        comp_model, comp_preprocess_fn, comp_postprocess_fn = get_model(
            model_name,
            model_config,
            model_checkpoint,
            args.lora,
            lora=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            lora_bias=args.lora_bias,
        )
        # comp_model = comp_model.to(device=device)
        if args.lora and callable(getattr(comp_model, "merge_and_unload", None)):
            comp_model.merge_and_unload()
        comp_model.eval()
        comparison_models.append((comp_model, comp_preprocess_fn, comp_postprocess_fn))

    # Dataset and Simulator
    _, _, test_loader = get_data(
        args.dataset,
        args.test_size,
        args.first_crop,
        args.split,
        args.batch_size,
        args.num_workers,
    )

    simulator = Microscope.from_file(args.microscope_config).to(device=device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device=device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)

    run_name = f"test_{args.main_model_name}_{'_'.join(Path(args.checkpoint).parts[2:])}".rstrip(
        "_"
    )

    wandb.init(
        project="XL-SIM",
        name=run_name,
    )
    total_parameters = sum(p.numel() for p in model.parameters())
    wandb.config.update(
        {
            "total_parameters": total_parameters,
        }
    )

    logged_batch = False

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test progress"):
            targets = batch["hr"].to(device=device)
            target_padding: torch.Tensor = batch["padding"]

            pixel_values, calibs = simulator(targets)
            pixel_values, targets, _ = crop_tensor(
                pixel_values,
                args.second_crop,
                pair_image=targets,
                pair_scale_factor=args.upscale,
                offset=target_padding.max(dim=0).values // (2 * args.upscale),
                mode="center",
            )
            psf = simulator.microscope.psf_em
            preprocessed_batch = preprocess_fn(
                pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale
            )

            outputs = model(**preprocessed_batch)
            outputs = postprocess_fn(outputs)

            psnr_fn.update(outputs, targets)
            ssim_fn.update(outputs, targets)

            if not logged_batch:
                comparison_samples = []
                for comp_model, comp_pre_fn, comp_post_fn in comparison_models:
                    comp_preprocessed = comp_pre_fn(
                        pixel_values=pixel_values,
                        psf=psf,
                        calibs=calibs,
                        upscale=args.upscale,
                    )
                    comp_out = comp_model(**comp_preprocessed)
                    comparison_samples.append(
                        (comp_post_fn(comp_out, target=targets), comp_model._get_name())
                    )

                images = []
                for i, (pred, inp, target) in enumerate(
                    zip(outputs, pixel_values, targets)
                ):
                    if i > 4:
                        break
                    nrow = 2 + len(comparison_samples)
                    tiles = [
                        target,
                        pred,
                        *[img[i] for img, _ in comparison_samples],
                        F.interpolate(
                            inp[None, 12:13], scale_factor=args.upscale, mode="nearest"
                        )[0],
                        torch.abs(pred - target),
                        *[torch.abs(img[i] - target) for img, _ in comparison_samples],
                    ]

                    image = make_grid(tiles, nrow=nrow)
                    images.append(
                        wandb.Image(
                            image.clip(0, 1),
                            caption=f"Sample {i}:\n Top left: Target | Top right: Prediction\n Bottom left: Input C12",
                        )
                    )
                wandb.log({"test/images": images})

                logged_batch = True

        metrics = {
            "psnr": psnr_fn.compute().item(),
            "ssim": ssim_fn.compute().item(),
        }

        wandb.log(metrics)

        print()
        print("==== Test Results ====")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

        wandb.finish()


if "__main__" == __name__:
    from configargparse import ArgumentParser

    parser = ArgumentParser()

    # Config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Models configuration
    parser.add_argument(
        "--main_model_name",
        type=str,
        default="Swin2SR",
        help="Name of the main model to test",
    )
    parser.add_argument(
        "--main_model_config",
        type=str,
        default="configs/models/swin2srX2.json",
        help="Path to the main model configuration",
    )
    parser.add_argument(
        "--comparison_model_names",
        type=str,
        nargs="+",
        default=["RL_Sum", "Sum"],
        help="Names of other methods to compare against",
    )
    parser.add_argument(
        "--comparison_model_configs",
        type=str,
        nargs="+",
        default=["configs/models/rlX2_sum.json", "configs/models/rlX2_sum.json"],
        help="Path to the other methods configuration. Must be in the same order as the names.",
    )
    parser.add_argument("--upscale", type=int, default=2, help="Upsampling factor")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--comparison_checkpoints",
        type=str,
        nargs="+",
        default=[None, None],
        help="Path to checkpoint to resume training",
    )

    # Dataset, preprocessing and postprocessing
    parser.add_argument(
        "--dataset", type=str, default="data/DIV2K", help="Path to the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="valid", help="Name of the split to load"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.99,
        help="Size of the validation/test set of the dataset",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Num workers for DataLoader"
    )
    parser.add_argument(
        "--first_crop", type=int, default=256, help="Size of LR first cropping"
    )
    parser.add_argument(
        "--second_crop", type=int, default=64, help="Size of LR second cropping"
    )
    parser.add_argument(
        "--lora", action="store_true", default=False, help="Whether to use lora or not"
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="Rank of the lora matrices"
    )
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha of the lora")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="Dropout rate of the lora"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["all-linear"],
        help="Layers to target with lora",
    )
    parser.add_argument(
        "--lora_bias", type=str, default="none", help="Bias to target with lora"
    )

    # Simulator
    parser.add_argument(
        "--microscope_config",
        type=str,
        default="configs/simulator/default_microscope.yaml",
        help="Path to the microscope configuration",
    )
    parser.add_argument(
        "--noise_config",
        type=str,
        default="configs/simulator/default_noise.yaml",
        help="Path to the noise configuration",
    )

    args = parser.parse_args()

    main(args)
