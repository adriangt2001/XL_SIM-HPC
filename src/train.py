import torch
from configargparse import Namespace

from .data.datasets import get_data
from .methods import get_model
from .training.losses import get_loss
from .training.trainer import Trainer
from .training.utils import get_optimizer, get_scheduler


def convert_memory_format(model: torch.nn.Module):
    for param in model.parameters():
        if param.ndim == 4:
            param.data = param.data.to(memory_format=torch.channels_last)
        elif param.ndim == 5:
            param.data = param.data.to(memory_format=torch.channels_last_3d)

    for buffer in model.buffers():
        if buffer.ndim == 4:
            buffer.data = buffer.data.to(memory_format=torch.channels_last)
        elif buffer.ndim == 5:
            buffer.data = buffer.data.to(memory_format=torch.channels_last_3d)

    return model


def main(args: Namespace):
    model, preprocess_fn, postprocess_fn = get_model(
        args.model_name,
        args.model_config,
        args.weights,
        False,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
    )
    model = convert_memory_format(model)

    train_loader, valid_loader, _ = get_data(
        args.dataset,
        args.test_size,
        args.first_crop,
        args.split,
        args.batch_size,
        args.num_workers,
    )
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        warmup_iterations=args.warmup_iterations,
        warmup_start_lr=args.warmup_lr,
        decay_iterations=args.decay_iterations,
        decay_factor=args.decay_factor,
    )
    loss = get_loss(
        args.l1_weight,
        args.wl1_weight,
        args.continuity_weight,
        args.sparsity_weight,
    )

    trainer = Trainer(
        model,
        preprocess_fn,
        postprocess_fn,
        loss,
        args.model_name,
        train_loader,
        valid_loader,
        valid_loader,
        optimizer,
        scheduler,
        args.microscope_config,
        args.noise_config,
        args.num_iterations,
        args.warmup_iterations,
        args.valid_freq,
        args.save_freq,
        args.output_dir,
        args.first_crop,
        args.second_crop,
        args.upscale,
        args.report_scalar_freq,
        args.report_image_freq,
        args.max_grad_norm,
        args.patience,
        checkpoint=args.checkpoint,
    )

    trainer.train()


if "__main__" == __name__:
    from configargparse import ArgumentParser

    parser = ArgumentParser()

    # Config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Model configuration
    parser.add_argument(
        "--model_name", type=str, default="Swin2SR", help="Name of the model to train"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/swin2srX2.json",
        help="Path to the model configuration",
    )
    parser.add_argument("--upscale", type=int, default=2, help="Upsampling factor")
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to model weights folder"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )

    # Dataset, preprocessing and postprocessing
    parser.add_argument(
        "--dataset", type=str, default="data/DIV2K", help="Path to the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Name of the split to load"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
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

    # Training
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=500000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw", help="Optimizer to use for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="multistep",
        help="Scheduler to use for the decay of LR",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=1000,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        help="Learning rate warmup initial factor",
    )
    parser.add_argument(
        "--decay_iterations",
        type=int,
        nargs="+",
        default=[250000, 400000, 450000, 475000],
        help="Iterations for the decay scheduler",
    )
    parser.add_argument(
        "--decay_factor",
        type=float,
        default=0.5,
        help="Decay factor for the learning rate scheduler",
    )
    parser.add_argument(
        "--loss_name", type=str, default="l1", help="Loss to use during training"
    )
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="+",
        default=[1.0],
        help="Weights for the loss terms (used when the loss has >1 terms)",
    )
    parser.add_argument(
        "--l1_weight",
        type=float,
        default=1.0,
        help="Weight of L1 loss in the final loss.",
    )
    parser.add_argument(
        "--wl1_weight",
        type=float,
        default=0.0,
        help="Weight of Weighted L1 loss in the final loss.",
    )
    parser.add_argument(
        "--continuity_weight",
        type=float,
        default=0.0,
        help="Weight of Continuity loss in the final loss.",
    )
    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=0.0,
        help="Weight of Sparsity loss in the final loss.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Validation iterations with no improvement to stop training",
    )
    parser.add_argument(
        "--valid_freq", type=int, default=1000, help="Iterations between validations"
    )
    parser.add_argument(
        "--save_freq", type=int, default=1000, help="Iterations between checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Path to checkpoints folder",
    )
    parser.add_argument(
        "--report_scalar_freq",
        type=int,
        default=1000,
        help="Iterations between scalar logs",
    )
    parser.add_argument(
        "--report_image_freq",
        type=int,
        default=1000,
        help="Iterations between image logs",
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
