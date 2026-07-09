from torch.nn import Module
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
)


def get_optimizer(model: Module, optimizer_name: str, lr: float) -> Optimizer:
    match optimizer_name:
        case "adamw":
            optimizer = AdamW(model.parameters(), lr)

        case "adam":
            optimizer = Adam(model.parameters(), lr)

        case _:
            raise ValueError(
                f"{optimizer_name} not implemented. Feel free to add it in utils.py."
            )

    return optimizer


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    decay_iterations: list[int],
    decay_factor: float,
    warmup_iterations: int,
    warmup_start_lr: float,
) -> SequentialLR:
    # Warmup Scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_lr,
        end_factor=1.0,
        total_iters=warmup_iterations,
    )

    # Decay Scheduler
    match scheduler_name:
        case "multistep":
            adjusted_iterations = [m - warmup_iterations for m in decay_iterations]
            decay_scheduler = MultiStepLR(
                optimizer, adjusted_iterations, gamma=decay_factor
            )

        case "cosine":
            decay_scheduler = CosineAnnealingLR(
                optimizer, decay_iterations[-1], eta_min=decay_factor
            )

        case "linear":
            decay_scheduler = LinearLR(
                optimizer, start_factor=1.0, end_factor=decay_factor
            )

        case _:
            raise ValueError(
                f"{scheduler_name} not implemented. Feel free to add it in utils.py."
            )

    # Mixed Scheduler
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, decay_scheduler], [warmup_iterations]
    )

    return scheduler
