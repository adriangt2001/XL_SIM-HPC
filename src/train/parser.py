from configargparse import ArgumentParser, Namespace


def parse_arguments(is_test: bool = False) -> Namespace:
    parser = ArgumentParser()

    if is_test:
        parser.add_argument("test", type=int, help="Test type")

    # Config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Model configuration
    parser.add_argument(
        "--model_name", type=str, default="Swin2SR", help="Name of the model to train"
    )
    parser.add_argument(
        "--in_num_channels", type=int, default=25, help="Number of input channels"
    )
    parser.add_argument(
        "--out_num_channels", type=int, default=1, help="Number of output channels"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Size of the patch window (only needed for transformer models)",
    )
    parser.add_argument("--upscale", type=int, default=2, help="Upsampling factor")

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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--scheduler_milestones",
        type=int,
        nargs="+",
        default=[250000, 400000, 450000, 475000],
        help="Milestones for MultiStepLR",
    )
    parser.add_argument(
        "--decay_factor",
        type=float,
        default=0.5,
        help="Decay factor for the learning rate scheduler",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
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

    args = parser.parse_args()

    return args
