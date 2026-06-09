from datasets import Dataset, Features, Image, Value
from pathlib import Path
import os
from tqdm import tqdm
from src.utils.read_mrc import read_mrc
import re
import numpy as np

data_root = Path("data/BioSR")

ccp_root = data_root / Path("CCPs")
er_root = data_root / Path("ER")
factin_root = data_root / Path("F-actin")
nlfactin_root = data_root / Path("F-actin_Nonlinear")
microtubules_root = data_root / Path("Microtubules")


# CCP, Microtubules, F-actin (linear)
gt_file = Path("SIM_gt.mrc")
lr_clean_file = Path("RawSIMData_gt.mrc")
for structure_root in (ccp_root, microtubules_root, factin_root):
    records = []
    highest_noise_level = 0
    for cell_folder in tqdm(sorted(structure_root.iterdir()), desc=f"Registering {structure_root.stem} data pairs"):
        if not cell_folder.is_dir():
            continue
        
        data_pairs = {"image_id": cell_folder.stem}
        hr_im = None
        for data_file in cell_folder.glob("*.mrc"):
            _, im = read_mrc(data_file)

            if data_file.stem == gt_file.stem:
                data_pairs["hr"] = im[..., 0]
                data_pairs["height"] = im.shape[0]
                data_pairs["width"] = im.shape[1]
                hr_im = im
            elif data_file.stem == lr_clean_file.stem:
                data_pairs["lr_x2"] = np.mean(im, axis=-1)
            else:
                noise_level = re.search(r'\d+', str(data_file.stem)).group()
                if int(noise_level) > highest_noise_level:
                    highest_noise_level = int(noise_level)
                data_pairs[f"lr_x2_{noise_level}"] = np.mean(im, axis=-1)
        records.append(data_pairs)
    features = Features(
        {
            "image_id": Value("string"),
            "hr": Image(),
            "lr_x2": Image(),
            **{f"lr_x2_{noise_level:02d}": Image() for noise_level in range(1, highest_noise_level + 1)},
            "height": Value("int32"),
            "width": Value("int32")
        }
    )
    dataset = Dataset.from_list(records, features=features)
    output_path = data_root / Path("data") / structure_root.stem
    os.makedirs(str(output_path), exist_ok=True)

    dataset.save_to_disk(str(output_path), max_shard_size="1GB")

# F-actin (Nonlinear)
gt_file1 = Path("SIM_gt_a.mrc")
gt_file2 = Path("SIM_gt_b.mrc")
lr_clean_file = Path("RawSIMData_gt.mrc")
records = []
highest_noise_level = 0
for cell_folder in tqdm(sorted(nlfactin_root.iterdir()), desc=f"Registering {nlfactin_root.stem} data pairs"):
    if not cell_folder.is_dir():
        continue
    
    data_pairs = {"image_id": cell_folder.stem}
    for data_file in cell_folder.glob("*.mrc"):
        _, im = read_mrc(data_file)

        if data_file.stem == gt_file1.stem:
            data_pairs["hr"] = np.mean(im, axis=-1)
            data_pairs["height"] = im.shape[0]
            data_pairs["width"] = im.shape[1]
        elif data_file.stem == gt_file2.stem:
            data_pairs["hr_detail"] = np.mean(im, axis=-1)
        elif data_file.stem == lr_clean_file.stem:
            data_pairs["lr_x3"] = np.mean(im, axis=-1)
        else:
            noise_level = re.search(r'\d+', str(data_file.stem)).group()
            if int(noise_level) > highest_noise_level:
                    highest_noise_level = int(noise_level)
            data_pairs[f"lr_x3_{noise_level}"] = np.mean(im, axis=-1)
    records.append(data_pairs)
features = Features(
    {
        "image_id": Value("string"),
        "hr": Image(),
        "hr_detail": Image(),
        "lr_x3": Image(),
        **{f"lr_x3_{noise_level:02d}": Image() for noise_level in range(1, highest_noise_level + 1)},
        "height": Value("int32"),
        "width": Value("int32")
    }
)
dataset = Dataset.from_list(records, features=features)
output_path = data_root / Path("data") / nlfactin_root.stem
os.makedirs(str(output_path), exist_ok=True)

dataset.save_to_disk(str(output_path), max_shard_size="1GB")

# # ER
# gt_folder = Path("GTSIM")
# lr_clean_folder = Path("RawGTSIMData")
# records = []
# for cell_folder in tqdm(sorted(er_root.iterdir()), desc=f"Registering {er_root.stem} data pairs"):
#     if not cell_folder.is_dir():
#         continue
    
#     # Get GT, Clean LR and Noisy LR

# # Review Features structure
# features = Features(
#     {
#         "image_id": Value("string"),
#         "hr": Value("string"),
#         "hr_detail": Value("string"),
#         "lr_x2": Value("string"),
#         **{f"lr_x2_{noise_level:02d}": Value("string") for noise_level in range(1, 10)},
#         "height": Value("int32"),
#         "width": Value("int32")
#     }
# )
# dataset = Dataset.from_list(records, features=features)
# output_path = data_root / Path("data") / er_root.stem
# os.makedirs(str(output_path), exist_ok=True)

# dataset.save_to_disk(str(output_path), max_shard_size="1GB")