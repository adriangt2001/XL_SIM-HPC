import torch
from configargparse import Namespace

from src.models import get_model

from .datasets import get_data
from .losses import get_loss
from .parser import parse_arguments_train
from .trainer import Trainer
from .utils import get_optimizer, get_scheduler


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
    model = model.to(memory_format=torch.channels_last)

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
        checkpoint=args.checkpoint,
    )

    trainer.train()


if "__main__" == __name__:
    args = parse_arguments_train()
    main(args)
