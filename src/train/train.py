import torch
from configargparse import Namespace
from torch.nn.functional import l1_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR#, MultiStepLR

from .datasets import prepare_data
from .models import get_model
from .parser import parse_arguments
from .trainer import Trainer


def main(args: Namespace):
    model, preprocess_fn, postprocess_fn = get_model(args)
    model = model.to(memory_format=torch.channels_last)
    train_loader, valid_loader = prepare_data(args)
    optimizer = AdamW(model.parameters(), args.lr)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_lr,
        end_factor=1.0,
        total_iters=args.warmup_iterations,
    )
    # adjusted_milestones = [
    #     m - args.warmup_iterations for m in args.scheduler_milestones
    # ]
    # decay_scheduler = MultiStepLR(
    #     optimizer, adjusted_milestones, gamma=args.decay_factor
    # )
    decay_scheduler = CosineAnnealingLR(optimizer, args.num_iterations, eta_min=1e-7)
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, decay_scheduler], [args.warmup_iterations]
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
    args = parse_arguments()
    main(args)
