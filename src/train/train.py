import os

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchvision.transforms.functional import crop

from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

import wandb
from src.simulation.microscope import Microscope
from src.simulation.sim_pipeline import ImageNoiseModel, SimulatorPipeline
from src.utils.datasets import Div2k


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        # self.model = torch.compile(model)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[gpu_id])
    
    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor, run: wandb.Run | None):
        self.optimizer.zero_grad()
        output = self.model(pixel_values=source).reconstruction
        loss = F.l1_loss(output, targets)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        if run is not None:
            run.log({"train/loss": loss.item(), "train/grad_norm": grad_norm.item()})
        
        return loss.item()
    
    def _run_epoch(self, epoch: int, run: wandb.Run | None):
        b_sz = self.train_data.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        avg_loss = 0
        count = 0
        if run is not None:
            with tqdm(self.train_data, desc=f"Epoch {epoch}") as pbar:
                for source, targets in pbar:
                    source = source.to(self.gpu_id, non_blocking=True)
                    targets = targets.to(self.gpu_id, non_blocking=True)
                    loss = self._run_batch(source, targets, run)
                    avg_loss += loss
                    count += 1
                    pbar.set_postfix({"loss": loss})    
        else:
            for source, targets in self.train_data:
                source = source.to(self.gpu_id, non_blocking=True)
                targets = targets.to(self.gpu_id, non_blocking=True)
                loss = self._run_batch(source, targets, run)
                avg_loss += loss
                count += 1
        avg_loss /= count

        if run is not None and self.gpu_id == 0:
            self.model.eval()
            with torch.no_grad():
                sample_src = source[0:1]
                sample_tgt = targets[0]
                
                pred = self.model(pixel_values=sample_src).reconstruction[0]
                pred = torch.clamp(pred, 0.0, 1.0)
                lr_vis = sample_src[0, 12:13] 
                
                # Log them as a group
                run.log({
                    "images/epoch": epoch,
                    "images/LR_Input_Ch12": wandb.Image(lr_vis, caption="Input (Channel 12)"),
                    "images/HR_Target": wandb.Image(sample_tgt, caption="Target"),
                    "images/Model_Prediction": wandb.Image(pred, caption="Prediction")
                })
            self.model.train()

        return avg_loss
    
    def _save_checkpoint(self, epoch: int):
        ckp = self.model.module.state_dict()
        PATH = "checkpoints/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def train(self, max_epochs: int):
        run = None
        if self.gpu_id == 0:
            run = wandb.init(project="DIV2K Experiments", name=self.model.module._get_name())
        
        best_loss = torch.finfo(torch.float32).max
        for epoch in range(max_epochs):
            avg_loss = self._run_epoch(epoch, run)
            if self.gpu_id == 0 and avg_loss < best_loss:
                self._save_checkpoint(epoch)
                best_loss = avg_loss
            # if self.gpu_id == 0 and (epoch + 1) % self.save_every == 0:
            #     self._save_checkpoint(epoch)
            
        if self.gpu_id == 0:
            run.finish()

def load_train_objs(dataset_path: str, rank: int):
    microscope = Microscope(resolution=(256, 256), device_id=rank)
    noise_model = ImageNoiseModel(device_id=rank)
    simulator = SimulatorPipeline(noise_model, microscope, device_id=rank)

    train_set = Div2k(dataset_path, target_size=512)

    config = Swin2SRConfig(image_size=64, num_channels=25, num_channels_out=1, window_size=8, upscale=2)
    model = Swin2SRForImageSuperResolution(config)
    print(f"Num parameters: {sum([p.numel() for p in model.parameters()])}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return train_set, model, optimizer, simulator

def prepare_dataloader(dataset: Dataset, batch_size: int, simulator: SimulatorPipeline):
    def my_collate_fn(samples):
        images, targets = zip(*samples)

        with torch.no_grad():
            simulated = [simulator(im.unsqueeze(0))[0][0] for im in images]
        
        simulated = torch.stack(simulated)
        targets = torch.stack(targets)

        top_lr = torch.randint(0, simulated.shape[2] - 64 + 1, (1,)).item()
        left_lr = torch.randint(0, simulated.shape[3] - 64 + 1, (1,)).item()
        top_gt = top_lr * 2
        left_gt = left_lr * 2

        simulated = crop(simulated, int(top_lr), int(left_lr), 64, 64)
        targets = crop(targets, int(top_gt), int(left_gt), 128, 128)

        return simulated, targets
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset), collate_fn=my_collate_fn)

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, dataset_path: str):
    ddp_setup(rank, world_size)
    try:
        dataset, model, optimizer, simulator = load_train_objs(dataset_path, rank)
        train_data = prepare_dataloader(dataset, batch_size, simulator)
        trainer = Trainer(model, train_data, optimizer, rank, save_every)
        trainer.train(total_epochs)
    finally:
        destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XL-SIM distributed training test")
    parser.add_argument('total_epochs', type=int, help="Total epochs to train the model")
    parser.add_argument('save_every', type=int, help="How often to save")
    parser.add_argument('dataset', type=str, help="Path to the dataset")
    parser.add_argument('--batch_size', default=32, type=int, help="Input batch size on each device (default: 32)")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args.dataset), nprocs=world_size)