import configargparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

import wandb
from src.simulation.microscope import Microscope
from src.simulation.sim_pipeline import ImageNoiseModel, SimulatorPipeline
from src.utils.datasets_large import Div2k
from src.utils.preprocessing import sr_random_crop


def ddp_setup():
    init_process_group(backend="nccl")  # Establish communication between processes
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    def __init__(
        self,
        local_rank: int,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        simulator: SimulatorPipeline,
        second_crop: int,
        upsampling: int,
        train_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        output_dir: str,
        num_iterations: int,
        save_every: int,
        report_scalar_rate: int,
        report_image_rate: int,
        max_grad_norm: float,
        checkpoint: str | None = None,
        compile_model: bool = False,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.model = model.to(local_rank, memory_format=torch.channels_last)
        self.model_name = self.model._get_name()
        if compile_model:
            self.model = torch.compile(self.model)
        self.model = DDP(self.model, device_ids=[local_rank])
        self.simulator = simulator
        self.second_crop = second_crop
        self.upsampling = upsampling
        self.train_data = train_data
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.num_iterations = num_iterations
        self.save_every = save_every
        self.report_scalar_rate = report_scalar_rate
        self.report_image_rate = report_image_rate
        self.max_grad_norm = max_grad_norm
        self.checkpoint = checkpoint

        if self.local_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

    def train_step(self, batch: dict[str, torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad()

        batch = {k: v.to(self.local_rank) for k, v in batch.items()}

        # Simulator & Cropping
        with torch.no_grad():
            pixel_values = self.simulator(batch["pixel_values"])[0]

        pixel_values, target = sr_random_crop(
            pixel_values, batch["target"], self.second_crop, self.second_crop * self.upsampling
        )

        proc_batch = {"pixel_values": pixel_values, "target": target}

        outputs = self.model(**proc_batch).reconstruction
        loss = F.l1_loss(outputs, proc_batch["target"])
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()
        return loss, grad_norm

    def train(self):
        if self.local_rank == 0:
            run = wandb.init(project="DIV2K Experiments", name=self.model_name)

        step = 0
        data_iterator = iter(self.train_data)

        # Overwrite if there is a checkpoint
        if self.checkpoint is not None:
            chkpt = torch.load(self.checkpoint)
            self.model.module.load_state_dict(chkpt["model_state_dict"])
            self.optimizer.load_state_dict(chkpt["optim_state_dict"])
            step = chkpt["step"]

        if self.local_rank == 0:
            pbar = tqdm(
                desc=f"Training {self.model_name}",
                total=self.num_iterations,
                initial=step,
            )

        # Full training loop
        while step < self.num_iterations:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_data)
                batch = next(data_iterator)

            loss, grad_norm = self.train_step(batch)

            # Log Scalar Values
            if step != 0 and step % self.report_scalar_rate == 0:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
                dist.reduce(grad_norm, dst=0, op=dist.ReduceOp.AVG)
                if self.local_rank == 0:
                    run.log(
                        {
                            "train/loss": loss.item(),
                            "train/grad_norm": grad_norm.item(),
                        },
                        step=step,
                    )
                    pbar.set_postfix({"loss": loss.item()})

            # Log Images
            if (
                self.local_rank == 0
                and step != 0
                and step % self.report_image_rate == 0
            ):
                self.model.eval()
                with torch.no_grad():
                    batch = {k: v.to(self.local_rank) for k, v in batch.items()}

                    # Simulator & Cropping
                    pixel_values = self.simulator(batch["pixel_values"])[0]
                    pixel_values, target = sr_random_crop(
                        pixel_values, batch["target"], self.second_crop, self.second_crop * self.upsampling
                    )

                    proc_batch = {"pixel_values": pixel_values, "target": target}

                    output = self.model(**proc_batch).reconstruction
                    output = torch.clamp(output, 0.0, 1.0)
                    run.log(
                        {
                            "images/HR_Prediction": wandb.Image(
                                output[0], caption="Prediction"
                            ),
                            "images/LR_Ch12": wandb.Image(
                                proc_batch["pixel_values"][0, 12:13], caption="Input (Channel 12)"
                            ),
                            "images/HR_Target": wandb.Image(
                                proc_batch["target"][0], caption="Target"
                            ),
                        },
                        step=step,
                    )

            if self.local_rank == 0 and step % self.save_every == 0:
                self.save_checkpoint(step + 1)

            if self.local_rank == 0:
                pbar.update(1)

            step += 1

    def save_checkpoint(self, step: int):
        model_state_dict = self.model.module.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        torch.save(
            {
                "model_name": self.model_name,
                "model_state_dict": model_state_dict,
                "optim_state_dict": optim_state_dict,
                "step": step,
            },
            os.path.join(self.output_dir, "checkpoint.pt"),
        )


def prepare_data(
    data_path: str,
    batch_size: int,
    num_workers: int,
    crop_size: int,
    upsampling: int,
):
    dataset = Div2k(data_path)

    def my_collate_fn(samples):
        sources, targets = zip(*samples)

        # Preprocessing 1
        proc_sources, proc_targets = [], []
        for source, target in zip(sources, targets):
            source, target = sr_random_crop(
                source.unsqueeze(0),
                target.unsqueeze(0),
                crop_size,
                crop_size * upsampling,
            )
            proc_sources.append(source[0])
            proc_targets.append(target[0])
        proc_sources = torch.stack(proc_sources)
        proc_targets = torch.stack(proc_targets)

        return {"pixel_values": proc_sources, "target": proc_targets}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=True),
        collate_fn=my_collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloader


def parse_arguments():
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--config', is_config_file=True, help="Path to config file")

    parser.add_argument('--dataset', type=str, default='data/DIV2K', help="Path to the dataset")
    parser.add_argument('--model_name', type=str, default='Swin2SR', help="Name of the model to train")
    parser.add_argument('--first_crop', type=int, default=256, help="Size of LR first cropping")
    parser.add_argument('--second_crop', type=int, default=64, help="Size of LR second cropping")
    parser.add_argument('--upsampling', type=int, default=2, help="Upsampling factor")
    parser.add_argument('--num_iterations', type=int, default=500000, help="Number of training iterations")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Num workers for DataLoader")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm fro clipping")
    parser.add_argument('--save_every', type=int, default=50, help="Iterations between checkpoints")
    parser.add_argument('--output_dir', type=str, default='checkpoints', help="Path to checkpoints folder")
    parser.add_argument('--report_scalar_rate', type=int, default=5, help="Iterations between scalar logs")
    parser.add_argument('--report_image_rate', type=int, default=25, help="Iterations between image logs")
    
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint to resume training")

    args = parser.parse_args()

    return args


def main():
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    ddp_setup()

    model_config = Swin2SRConfig(
        image_size=args.second_crop,
        num_channels=25,
        num_channels_out=1,
        window_size=8,
        upscale=args.upsampling,
    )

    model = Swin2SRForImageSuperResolution(model_config)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(f"Num parameters: {sum([p.numel() for p in model.parameters()])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = prepare_data(
        args.dataset,
        args.batch_size,
        args.num_workers,
        args.first_crop,
        args.upsampling,
    )

    sim_pipeline = SimulatorPipeline(
        noise=ImageNoiseModel(device_id=int(os.environ["LOCAL_RANK"])),
        microscope=Microscope(device_id=int(os.environ["LOCAL_RANK"]), resolution=(args.first_crop, args.first_crop)),
        device_id=int(os.environ["LOCAL_RANK"]),
    )

    trainer = Trainer(
        local_rank=int(os.environ["LOCAL_RANK"]),
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        model=model,
        simulator=sim_pipeline,
        second_crop=args.second_crop,
        upsampling=args.upsampling,
        train_data=train_dataloader,
        optimizer=optimizer,
        output_dir=os.path.join(args.output_dir, args.model_name),
        num_iterations=args.num_iterations,
        save_every=args.save_every,
        report_scalar_rate=args.report_scalar_rate,
        report_image_rate=args.report_image_rate,
        max_grad_norm=args.max_grad_norm,
        checkpoint=args.checkpoint,
        # compile_model=True,
    )

    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    main()
