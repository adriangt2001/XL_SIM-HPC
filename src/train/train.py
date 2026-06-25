from configargparse import Namespace
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR

from .datasets import prepare_data
from .models import get_model
from .parser import parse_arguments
from .trainer import Trainer
import torch


def main(args: Namespace):
    model, postprocess_fn = get_model(args)
    model = model.to(memory_format=torch.channels_last)
    train_loader, valid_loader = prepare_data(args)
    optimizer = Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_lr,
        end_factor=1.0,
        total_iters=args.warmup_iterations,
    )
    adjusted_milestones = [
        m - args.warmup_iterations for m in args.scheduler_milestones
    ]
    decay_scheduler = MultiStepLR(
        optimizer, adjusted_milestones, gamma=args.decay_factor
    )
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, decay_scheduler], [args.warmup_iterations]
    )

    trainer = Trainer(
        model,
        postprocess_fn,
        l1_loss,
        args.model_name,
        train_loader,
        valid_loader,
        valid_loader,
        optimizer,
        scheduler,
        args.num_iterations,
        args.warmup_iterations,
        args.valid_freq,
        args.save_freq,
        args.output_dir,
        args.first_crop,
        args.second_crop,
        args.upscale,
        args.report_scalar_freq,
        args.max_grad_norm,
        checkpoint=args.checkpoint,
    )

    trainer.train()


if "__main__" == __name__:
    args = parse_arguments()
    main(args)
