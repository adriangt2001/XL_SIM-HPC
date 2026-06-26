import datetime
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, broadcast_object_list, tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

import wandb
from src.simulation.sim_pipeline import SimulatorPipeline
from src.utils.preprocessing import sr_random_crop_tensor


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess_fn: Callable[[any], dict[str, torch.Tensor]],
        postprocess_fn: Callable[[any], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_name: str,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        microscope_filename: str,
        noise_filename: str,
        max_iters: int,
        warmup_iters: int,
        valid_freq: int,
        save_freq: int,
        output_dir: str,
        first_crop_size: int,
        second_crop_size: int,
        upscale: int,
        log_freq: int,
        image_log_freq: int,
        max_grad_norm: float,
        checkpoint: str | None = None,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)

        self.accelerator = Accelerator()
        self.model_name = model_name
        self.output_dir = [None]
        if self.accelerator.is_main_process:
            self.output_dir[0] = self.__generate_run_path(output_dir, model_name)
        broadcast_object_list(self.output_dir, from_process=0)
        self.output_dir = self.output_dir[0]
        self.run_name = f"{model_name}_{self.output_dir.name}"

        self.accelerator = Accelerator(
            project_config=ProjectConfiguration(
                project_dir=str(self.output_dir),
                automatic_checkpoint_naming=True,
                total_limit=5,
            ),
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs],
            step_scheduler_with_optimizer=False,
        )
        self.accelerator.init_trackers(
            project_name="XL-SIM",
            init_kwargs={
                "wandb": {
                    "name": self.run_name,
                    "dir": str(self.output_dir),
                }
            },
        )
        self.device = self.accelerator.device

        (
            self.model,
            self.train_loader,
            self.valid_loader,
            self.test_loader,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, train_loader, valid_loader, test_loader, optimizer, scheduler
        )
        self.accelerator.register_for_checkpointing(self.scheduler)

        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.valid_freq = valid_freq
        self.save_freq = save_freq
        self.first_crop_size = first_crop_size
        self.second_crop_size = second_crop_size
        self.upscale = upscale
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.loss_fn = loss_fn
        self.log_freq = log_freq
        self.image_log_freq = image_log_freq
        self.max_grad_norm = max_grad_norm
        self.simulator = SimulatorPipeline.from_file(
            microscope_filename, noise_filename
        ).to(device=self.device)
        self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device=self.device)
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            device=self.device
        )

        self.checkpoint = None
        if checkpoint is not None:
            self.checkpoint = Path(checkpoint)

    def __generate_run_path(self, output_dir, model_name):
        model_folder = Path(output_dir) / Path(model_name)
        model_folder.mkdir(exist_ok=True)

        id_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = Path(f"run_{id_run}")

        full_path = model_folder / run_folder

        full_path.mkdir(parents=True, exist_ok=True)

        return full_path

    def train_step(self, batch: dict[str, torch.Tensor]):
        self.optimizer.zero_grad(set_to_none=True)

        targets = batch["hr"]

        with torch.no_grad():
            pixel_values, calibs = self.simulator(targets)
            pixel_values, targets = sr_random_crop_tensor(
                pixel_values,
                targets,
                self.second_crop_size,
                self.second_crop_size * self.upscale,
            )

            preprocessed_batch = self.preprocess_fn(
                pixel_values=pixel_values, calibs=calibs
            )

        with self.accelerator.autocast():
            outputs = self.model(**preprocessed_batch)
            outputs = self.postprocess_fn(outputs)
            loss = self.loss_fn(outputs, targets)

        self.accelerator.backward(loss)

        self.accelerator.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )

        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss, lr

    def train(self):
        step = 0
        epoch = 0
        best_psnr = {"step": 0, "psnr": 0, "ssim": 0}
        best_ssim = {"step": 0, "psnr": 0, "ssim": 0}
        total_loss = 0
        self.model.train()

        if self.checkpoint is not None:
            step, epoch, best_psnr, best_ssim = self.load_state()

        pbar = tqdm(
            desc="Training progress",
            total=self.max_iters,
            initial=step,
            main_process_only=True,
        )
        while step < self.max_iters:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            for train_data in self.train_loader:
                loss, current_lr = self.train_step(train_data)
                loss = self.accelerator.reduce(loss, reduction="mean").item()
                total_loss += loss
                step += 1

                pbar.update(1)
                pbar.set_postfix({"loss": loss})

                if step % self.log_freq == 0:
                    self.log_train(total_loss / self.log_freq, current_lr, step)
                    total_loss = 0

                if step >= self.warmup_iters and step % self.valid_freq == 0:
                    loss, metrics, predictions, pixel_values, targets = self.valid_step(
                        self.valid_loader
                    )
                    self.log_valid(
                        loss,
                        metrics,
                        predictions,
                        pixel_values,
                        targets,
                        step,
                        split="valid",
                    )

                    if metrics["psnr"] > best_psnr["psnr"]:
                        best_psnr["step"] = step
                        best_psnr["psnr"] = metrics["psnr"]
                        best_psnr["ssim"] = metrics["ssim"]
                        self.save_model("best_psnr")

                    if metrics["ssim"] > best_ssim["ssim"]:
                        best_ssim["step"] = step
                        best_ssim["psnr"] = metrics["psnr"]
                        best_ssim["ssim"] = metrics["ssim"]
                        self.save_model("best_ssim")

                if step % self.save_freq == 0:
                    self.save_state(step, epoch, best_psnr, best_ssim)

                if step >= self.max_iters:
                    break

            epoch += 1
        self.accelerator.end_training()

    @torch.inference_mode()
    def valid_step(self, loader):
        self.model.eval()

        self.psnr_fn.reset()
        self.ssim_fn.reset()

        total_loss = 0
        count = 0
        for batch in tqdm(loader, desc="Valid progress", main_process_only=True):
            targets = batch["hr"]

            pixel_values, calibs = self.simulator(targets)
            pixel_values = pixel_values[
                ..., : self.second_crop_size, : self.second_crop_size
            ]
            targets = targets[
                ..., : self.second_crop_size * 2, : self.second_crop_size * 2
            ]
            preprocessed_batch = self.preprocess_fn(
                pixel_values=pixel_values, calibs=calibs
            )

            with self.accelerator.autocast():
                outputs = self.model(**preprocessed_batch)
                outputs = self.postprocess_fn(outputs)
                loss = self.loss_fn(outputs, targets)

            total_loss += loss.item()
            count += 1

            gathered_outputs, gathered_targets = self.accelerator.gather_for_metrics(
                (outputs, targets)
            )
            self.psnr_fn.update(gathered_outputs, gathered_targets)
            self.ssim_fn.update(gathered_outputs, gathered_targets)

        local_avg_loss = torch.tensor(total_loss / count, device=self.device)
        global_avg_loss = self.accelerator.reduce(
            local_avg_loss, reduction="mean"
        ).item()

        metrics = {
            "psnr": self.psnr_fn.compute().item(),
            "ssim": self.ssim_fn.compute().item(),
        }

        self.model.train()

        return global_avg_loss, metrics, outputs, pixel_values, targets

    def save_model(self, name: str):
        self.accelerator.save_model(self.model, self.output_dir / name)

    def save_state(
        self,
        step: int,
        epoch: int,
        best_psnr: dict[str, int],
        best_ssim: dict[str, int],
    ):
        self.accelerator.save_state(self.output_dir)
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
            },
            self.output_dir / Path("train_info.pt"),
        )

    def load_state(self):
        self.accelerator.load_state(self.checkpoint.parts[:-1])
        ckpt = torch.load(self.checkpoint / Path("train_info.pt"))
        step = ckpt["step"]
        epoch = ckpt["epoch"]
        best_psnr = ckpt["best_psnr"]
        best_ssim = ckpt["best_ssim"]
        return step, epoch, best_psnr, best_ssim

    def log_train(self, loss: int, lr: float, step: int):
        self.accelerator.log(
            {"train/loss": loss, "train/lr": lr},
            step=step,
        )

    def log_valid(
        self,
        loss: int,
        metrics: dict[str, float],
        predictions: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
        split: str = "valid",
    ):
        self.accelerator.log(
            {
                f"{split}/loss": loss,
                f"{split}/psnr": metrics["psnr"],
                f"{split}/ssim": metrics["ssim"],
            },
            step=step,
        )

        if self.accelerator.is_main_process and step % self.image_log_freq == 0:
            images = []
            for i, (pred, inp, target) in enumerate(zip(predictions, inputs, targets)):
                if i > 4:
                    break
                image = make_grid(
                    [
                        target,
                        pred,
                        F.interpolate(
                            inp[None, 12:13], scale_factor=args.upscale, mode="nearest"
                        )[0],
                    ],
                    nrow=2,
                )
                images.append(
                    wandb.Image(
                        image.clip(0, 1),
                        caption=f"Sample {i}:\n Top left: Target | Top right: Prediction\n Bottom left: Input C12",
                    )
                )
            wandb.log(
                {f"{split}/images": images},
                step=step,
            )


if "__main__" == __name__:
    from src.train.datasets import prepare_data
    from src.train.models import get_model
    from src.train.parser import parse_arguments

    args = parse_arguments(is_test=True)

    model = get_model(args)
    train_loader, valid_loader = prepare_data(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
    output_dir = args.output_dir

    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        valid_loader,
        optimizer,
        scheduler,
        output_dir,
    )
