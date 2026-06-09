from datasets import Dataset, Image, Features, Value
from pathlib import Path
from PIL import Image as IM
import os
from tqdm import tqdm

# Training set
hr_root = Path("data/LSDIR/train_hr")
lr_root = Path("data/LSDIR/train_lr")

records = []

for hr_folder in tqdm(sorted(hr_root.iterdir()), desc="Registering data pairs"):
    if not hr_folder.is_dir():
        continue

    lr_folder = lr_root / hr_folder.name

    for hr_img in hr_folder.glob("*.png"):
        image_id = hr_img.stem
        
        im = IM.open(hr_img)

        records.append(
            {
                "image_id": image_id,
                "hr": str(hr_img),
                "lr_x2": str(lr_folder / f"{image_id}x2.png"),
                "lr_x3": str(lr_folder / f"{image_id}x3.png"),
                "lr_x4": str(lr_folder / f"{image_id}x4.png"),
                "height": im.height,
                "width": im.width,
            }
        )

features = Features(
    {
        "image_id": Value("string"),
        "hr": Image(),
        "lr_x2": Image(),
        "lr_x3": Image(),
        "lr_x4": Image(),
        "height": Value("int32"),
        "width": Value("int32"),
    }
)

dataset = Dataset.from_list(records, features=features, split="train")
output_path = "data/LSDIR/data/"
os.makedirs(output_path, exist_ok=True)

dataset.save_to_disk(output_path, num_shards=200)


# Validation set
dataset_root = Path("data/LSDIR/val1")
hr_root = dataset_root / Path("HR")

records = []

for hr_folder in tqdm(sorted(hr_root.iterdir()), desc="Registering data pairs"):
    if not hr_folder.is_dir():
        continue

    for hr_img in hr_folder.glob("*.png"):
        image_id = hr_img.stem
        
        im = IM.open(hr_img)

        records.append(
            {
                "image_id": image_id,
                "hr": str(hr_img),
                "lr_x2": str(dataset_root / Path("X2/val") / f"{image_id}x2.png"),
                "lr_x3": str(dataset_root / Path("X3/val") / f"{image_id}x3.png"),
                "lr_x4": str(dataset_root / Path("X4/val") / f"{image_id}x4.png"),
                "height": im.height,
                "width": im.width,
            }
        )

features = Features(
    {
        "image_id": Value("string"),
        "hr": Image(),
        "lr_x2": Image(),
        "lr_x3": Image(),
        "lr_x4": Image(),
        "height": Value("int32"),
        "width": Value("int32"),
    }
)

dataset = Dataset.from_list(records, features=features, split="validation")
output_path = "data/LSDIR/data/validation"
os.makedirs(output_path, exist_ok=True)

dataset.save_to_disk(output_path, max_shard_size="1GB")
