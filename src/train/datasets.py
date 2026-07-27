import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)
from torchvision.transforms.functional import to_tensor

from src.utils.preprocessing import crop_pil


def _prepare_biosr(data_path: Path, test_size: float, first_crop: int):
    assert test_size < 0.5, (
        f"BioSR dataset assumes validation and test size are equal. Current test_size is {test_size}. Make sure it is <0.5."
    )
    data_path = data_path / Path("data")

    datasets = []
    for split in sorted(data_path.iterdir()):
        ds = load_from_disk(str(split))
        ds = ds.add_column("class", [split.name] * len(ds))
        datasets.append(ds)

    dataset: Dataset = concatenate_datasets(datasets)
    dataset = dataset.class_encode_column("class")

    splits1 = dataset.train_test_split(
        test_size=test_size * 2, shuffle=True, stratify_by_column="class", seed=42
    )
    train_dataset = splits1["train"]
    test_dataset = splits1["test"]

    splits2 = test_dataset.train_test_split(
        test_size=0.5, shuffle=True, stratify_by_column="class", seed=42
    )
    valid_dataset = splits2["train"]
    test_dataset = splits2["test"]
    transforms = Compose(
        [
            RandomRotation(degrees=180),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]
    )

    def transform_train(sample):
        hrs = []
        padding = []

        for idx in range(len(sample["hr"])):
            hr = sample["hr"][idx]
            hr = transforms(hr)

            hr, _, pad = crop_pil(hr, first_crop, mode="random")
            hr = np.array(hr) / 65565
            hr = torch.from_numpy(hr)[None, ...]

            hrs.append(hr)
            padding.append(torch.as_tensor(pad))

        return {"hr": hrs, "padding": padding}

    def transform_test(sample):
        hrs = []
        padding = []

        for idx in range(len(sample["hr"])):
            hr = sample["hr"][idx]
            hr, _, pad = crop_pil(hr, first_crop, mode="center")
            hr = np.array(hr) / 65565
            hr = torch.from_numpy(hr)[None, ...]
            hrs.append(hr)
            padding.append(torch.as_tensor(pad))

        return {"hr": hrs, "padding": padding}

    train_dataset.set_transform(transform_train)
    valid_dataset.set_transform(transform_test)
    test_dataset.set_transform(transform_test)

    def my_collate_fn(batch):
        return {
            "hr": torch.stack([sample["hr"] for sample in batch]),
            "padding": torch.stack([sample["padding"] for sample in batch]),
        }

    return train_dataset, valid_dataset, test_dataset, my_collate_fn


def _prepare_lsdir(data_path: Path, test_size: float, first_crop: int, split: str):
    dataset = load_from_disk(str(data_path / Path("data", split)))

    splits = dataset.train_test_split(test_size=test_size, shuffle=False)
    train_dataset = splits["train"]
    valid_dataset = splits["test"]
    transforms = Compose(
        [
            RandomRotation(degrees=180),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]
    )

    def transform_train(sample):
        hrs = []
        padding = []

        for idx in range(len(sample["hr"])):
            hr = sample["hr"][idx].convert("L")
            hr = transforms(hr)

            hr, _, pad = crop_pil(hr, first_crop, mode="random")
            hr = to_tensor(hr)

            hrs.append(hr)
            padding.append(torch.as_tensor(pad))

        return {"hr": hrs, "padding": padding}

    def transform_test(sample):
        hrs = []
        padding = []

        for idx in range(len(sample["hr"])):
            hr = sample["hr"][idx].convert("L")
            hr, _, pad = crop_pil(hr, first_crop, mode="center")
            hrs.append(to_tensor(hr))
            padding.append(torch.as_tensor(pad))

        return {"hr": hrs, "padding": padding}

    train_dataset.set_transform(transform_train)
    valid_dataset.set_transform(transform_test)

    def my_collate_fn(batch):
        return {
            "hr": torch.stack([sample["hr"] for sample in batch]),
            "padding": torch.stack([sample["padding"] for sample in batch]),
        }

    return train_dataset, valid_dataset, valid_dataset, my_collate_fn


def get_data(
    dataset: str,
    test_size: float,
    first_crop: int,
    split: str,
    batch_size: int,
    num_workers: int,
):
    data_path = Path(dataset)

    match data_path.stem:
        case "LSDIR":
            train_dataset, valid_dataset, test_dataset, my_collate_fn = _prepare_lsdir(
                data_path, test_size, first_crop, split
            )

        case "BioSR":
            train_dataset, valid_dataset, test_dataset, my_collate_fn = _prepare_biosr(
                data_path, test_size, first_crop
            )

        case _:
            raise ValueError(
                f"{data_path.stem} dataset not implemented. Feel free to add it in datasets.py."
            )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def prepare_data_distributed(
    dataset: str,
    test_size: float,
    first_crop: int,
    split: str,
    batch_size: int,
    num_workers: int,
):
    data_path = Path(dataset)

    match data_path.stem:
        case "LSDIR":
            train_dataset, valid_dataset, my_collate_fn = _prepare_lsdir(
                data_path, test_size, first_crop, split
            )
        case _:
            raise ValueError(
                f"{data_path.stem} dataset not implemented. Feel free to add it in datasets.py."
            )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset, shuffle=True),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=DistributedSampler(valid_dataset, shuffle=False),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=my_collate_fn,
        prefetch_factor=2,
    )

    return train_dataloader, valid_dataloader


if "__main__" == __name__:
    from tqdm import tqdm

    from src.train.parser import parse_arguments_train

    args = parse_arguments_train(is_test=True)

    if 1 == args.test:
        train_dataloader, valid_dataloader = get_data(args)
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

        train_dataloader, valid_dataloader = get_data(args)
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
