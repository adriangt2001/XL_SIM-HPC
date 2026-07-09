import torch
from configargparse import Namespace
from torch.nn.functional import l1_loss
from torch.optim import AdamW

from src.models import get_model

from .datasets import prepare_data
from .parser import parse_arguments_train
from .utils import get_scheduler, get_optimizer
from .trainer import Trainer


def main(args: Namespace):
    model, preprocess_fn, postprocess_fn = get_model(
        args.model_name, args.model_config, args.checkpoint
    )
    model = model.to(memory_format=torch.channels_last)
    train_loader, valid_loader = prepare_data(args)
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        warmup_iterations=args.warmup_iterations,
        warmup_start_lr=args.warmup_lr,
        decay_iterations=args.decay_iterations,
        decay_factor=args.decay_factor,
    )

    trainer = Trainer(
        model,
        preprocess_fn,
        postprocess_fn,
        l1_loss,
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
        checkpoint=args.checkpoint,
    )

    trainer.train()


if "__main__" == __name__:
    args = parse_arguments_train()
    main(args)
