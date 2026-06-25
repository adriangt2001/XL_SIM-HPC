import os
from pathlib import Path

import torch
import torch.distributed as dist
from configargparse import Namespace
from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.functional import to_tensor

from src.utils.preprocessing import random_crop_pil, crop_pil


def _prepare_lsdir(data_path: Path, args: Namespace):
    dataset = load_from_disk(str(data_path / Path("data", args.split)))

    splits = dataset.train_test_split(test_size=args.test_size, shuffle=False)
    train_dataset = splits["train"]
    valid_dataset = splits["test"]

    def transform_train(sample):
        hrs = []

        for idx in range(len(sample["hr"])):
            hr = sample["hr"][idx].convert('L')
            hr = random_crop_pil(
                hr,
                args.first_crop,
            )
            hrs.append(hr)

        return {
            "hr": [to_tensor(im) for im in hrs]
        }
    
    def transform_test(sample):
        hrs = []

        for idx in range(len(sample["hr"])):
            hr = sample["hr"][idx].convert('L')
            hr = crop_pil(hr, args.first_crop)
            hrs.append(hr)
        
        return {
            "hr": [to_tensor(im) for im in hrs]
        }

    train_dataset.set_transform(transform_train)
    valid_dataset.set_transform(transform_test)

    def my_collate_fn(batch):
        return {
            "hr": torch.stack([sample["hr"] for sample in batch]),
        }

    return train_dataset, valid_dataset, my_collate_fn


def prepare_data(args: Namespace):
    data_path = Path(args.dataset)

    if "LSDIR" == data_path.stem:
        train_dataset, valid_dataset, my_collate_fn = _prepare_lsdir(data_path, args)
    else:
        pass

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    return train_dataloader, valid_dataloader


def prepare_data_distributed(args: Namespace):
    data_path = Path(args.dataset)

    if "LSDIR" == data_path.stem:
        train_dataset, valid_dataset, my_collate_fn = _prepare_lsdir(data_path, args)
    else:
        pass

    train_dataloader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset, shuffle=True),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=DistributedSampler(valid_dataset, shuffle=False),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    return train_dataloader, valid_dataloader


if "__main__" == __name__:
    from tqdm import tqdm

    from src.train.parser import parse_arguments

    args = parse_arguments(is_test=True)

    if 1 == args.test:
        train_dataloader, valid_dataloader = prepare_data(args)
        train_dataset = train_dataloader.dataset
        valid_dataset = valid_dataloader.dataset
        print("==== Train Dataset ====")
        print(f"Number of samples: {train_dataset}")
        print(f"Dataset columns: {train_dataset[0].keys()}")
        print(f"HR analysis: {train_dataset[0]['hr']}")
        print(f"LR analysis: {train_dataset[0]['lr']}")
        print()

        print("==== Valid Dataset ====")
        print(f"Number of samples: {valid_dataset}")
        print(f"Dataset columns: {valid_dataset[0].keys()}")
        print(f"HR analysis: {valid_dataset[0]['hr']}")
        print(f"LR analysis: {valid_dataset[0]['lr']}")
        print()

        mean_size = 0
        count = 0
        for batch in tqdm(train_dataloader):
            mean_size += batch["hr"].shape[2] * batch["hr"].shape[3]
            count += 1
        print(f"Mean size: {mean_size // count}")

    elif 2 == args.test:
        from torch.distributed import destroy_process_group, init_process_group

        def ddp_setup():
            init_process_group(
                backend="nccl"
            )  # Establish communication between processes
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        ddp_setup()

        train_dataloader, valid_dataloader = prepare_data_distributed(args)
        train_dataset = train_dataloader.dataset
        valid_dataset = valid_dataloader.dataset
        print("==== Train Dataset ====")
        print(f"Number of samples: {train_dataset}")
        print(f"Dataset columns: {train_dataset[0].keys()}")
        print(f"HR analysis: {train_dataset[0]['hr']}")
        print(f"LR analysis: {train_dataset[0]['lr']}")
        print()

        print("==== Valid Dataset ====")
        print(f"Number of samples: {valid_dataset}")
        print(f"Dataset columns: {valid_dataset[0].keys()}")
        print(f"HR analysis: {valid_dataset[0]['hr']}")
        print(f"LR analysis: {valid_dataset[0]['lr']}")
        print()

        mean_size = 0
        count = 0
        for batch in tqdm(train_dataloader):
            mean_size += batch["hr"].shape[2] * batch["hr"].shape[3]
            count += 1
        mean_size = torch.tensor(mean_size, device="cuda")
        count = torch.tensor(count, device="cuda")
        dist.all_reduce(mean_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        print(f"Mean size: {mean_size.item() // count.item()}")

        destroy_process_group()
    elif 3 == args.test:
        from accelerate import Accelerator

        accelerator = Accelerator()
        device = accelerator.device

        train_dataloader, valid_dataloader = prepare_data(args)
        train_dataset = train_dataloader.dataset
        valid_dataset = valid_dataloader.dataset
        print("==== Train Dataset ====")
        print(f"Number of samples: {train_dataset}")
        print(f"Dataset columns: {train_dataset[0].keys()}")
        print(f"HR analysis: {train_dataset[0]['hr']}")
        print(f"LR analysis: {train_dataset[0]['lr']}")
        print()

        print("==== Valid Dataset ====")
        print(f"Number of samples: {valid_dataset}")
        print(f"Dataset columns: {valid_dataset[0].keys()}")
        print(f"HR analysis: {valid_dataset[0]['hr']}")
        print(f"LR analysis: {valid_dataset[0]['lr']}")
        print()

        train_dataloader = accelerator.prepare(train_dataloader)

        mean_size = 0
        count = 0
        for batch in tqdm(train_dataloader, disable=not accelerator.is_main_process):
            mean_size += batch["hr"].shape[2] * batch["hr"].shape[3]
            count += 1
        mean_size = torch.tensor(mean_size, device=device)
        count = torch.tensor(count, device=device)

        accelerator.reduce(mean_size, reduction="sum")
        accelerator.reduce(count, reduction="sum")
        accelerator.print(f"Mean size: {mean_size.item() // count.item()}")

        accelerator.end_training()
