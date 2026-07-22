from configargparse import ArgumentParser, Namespace


def parse_arguments_train(is_test: bool = False) -> Namespace:
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

    return args


def parse_arguments_test(is_test: bool = False) -> Namespace:
    parser = ArgumentParser()

    if is_test:
        parser.add_argument("test", type=int, help="Test type")

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
        "--split", type=str, default="train", help="Name of the split to load"
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

    return args
