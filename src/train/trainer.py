from collections.abc import Callable
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, tqdm
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.simulation.sim_pipeline import SimulatorPipeline
from src.utils.preprocessing import sr_random_crop_pil


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        postprocess_fn: Callable[[any], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_name: str,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        max_iters: int,
        valid_freq: int,
        save_freq: int,
        output_dir: str,
        crop_size: int,
        upscale: int,
        checkpoint: str | None = None,
    ):
        self.model_name = model_name
        self.output_dir = self.__generate_run_path(output_dir, model_name)
        self.run_name = f"{model_name}_{self.output_dir.name}"

        self.accelerator = Accelerator(
            project_config=ProjectConfiguration(
                project_dir=str(self.output_dir),
                automatic_checkpoint_naming=True,
                total_limit=5,
            ),
            log_with="wandb",
        )
        self.accelerator.init_trackers(
            project_name=self.model_name,
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

        self.simulator = SimulatorPipeline(device=self.device)
        self.max_iters = max_iters
        self.valid_freq = valid_freq
        self.save_freq = save_freq
        self.crop_size = crop_size
        self.upscale = upscale
        self.postprocess_fn = postprocess_fn
        self.loss_fn = loss_fn
        self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device=self.device)
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            device=self.device
        )

        self.checkpoint = None
        if checkpoint is not None:
            self.checkpoint = Path(checkpoint)

    def __generate_run_path(self, model_name, output_dir):
        model_folder = Path(output_dir) / Path(model_name)

        num_runs = len(model_folder.iterdir())
        run_folder = Path(f"run_{num_runs:03d}")

        full_path = model_folder / run_folder

        full_path.mkdir(exist_ok=True)

        return full_path

    def train_step(self, batch: dict[str, torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pixel_values = batch["lr"]
        target = batch["hr"]

        with torch.no_grad():
            pixel_values, _ = self.simulator(pixel_values)

        pixel_values, target = sr_random_crop_pil(
            pixel_values, target, self.crop_size, self.crop_size * self.upscale
        )

        with self.accelerator.autocast():
            outputs = self.model(pixel_values)
            outputs = self.postprocess_fn(outputs)
            loss = self.loss_fn(outputs, target)

        self.accelerator.backward(loss)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def train(self):
        step = 0
        epoch = 0
        best_psnr = {"step": 0, "psnr": 0, "ssim": 0}
        best_ssim = {"step": 0, "psnr": 0, "ssim": 0}
        total_loss = 0

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
                loss = self.train_step(train_data)
                total_loss += loss
                step += 1

                pbar.update(1)
                pbar.set_postfix({"loss": loss})

                if step % self.log_freq == 0:
                    self.log_train(total_loss / self.log_freq)
                    total_loss = 0

                if step % self.valid_freq == 0:
                    loss, metrics = self.valid_step(self.valid_loader)
                    self.log_valid(loss, metrics)

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

    @torch.no_grad()
    def valid_step(self, loader):
        self.model.eval()
        metrics = {"psnr": 0, "ssim": 0}
        total_loss = 0
        count = 0
        for batch in tqdm(loader, desc="Valid progress", main_process_only=True):
            pixel_values = batch["lr"]
            targets = batch["hr"]

            pixel_values, _ = self.simulator(pixel_values)
            outputs = self.model(pixel_values)
            outputs = self.postprocess_fn(outputs)
            loss = self.loss_fn(outputs, targets)

            total_loss += loss.item()
            metrics["psnr"] += self.psnr_fn(outputs, targets)
            metrics["ssim"] += self.ssim_fn(outputs, targets)
            count += 1

        total_loss /= count
        metrics["psnr"] /= count
        metrics["ssim"] /= count

        return total_loss, metrics

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
        best_psnr = ckpt["best_psnr"]
        best_ssim = ckpt["best_ssim"]
        return step, best_psnr, best_ssim

    def log_train(self, loss: int, step: int):
        self.accelerator.log(
            {"train/loss": loss},
            step=step,
        )

    def log_valid(
        self,
        loss: int,
        metrics: dict[str, float],
        predictions: torch.Tensor,
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

        if self.accelerator.is_main_process:
            pred_images = []
            target_images = []
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                pred_images.append(wandb.Image(pred, caption=f"Sample {i}: Prediction"))
                target_images.append(wandb.Image(target, caption=f"Sample {i}: Target"))
            wandb.log(
                {
                    f"{split}/predictions": pred_images,
                    f"{split}/targets": target_images,
                },
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
