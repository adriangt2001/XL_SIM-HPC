import glob
import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


class Div2k(torch.utils.data.Dataset):
    def __init__(self, source_path: str):
        super().__init__()
        self.gt_list, self.lr_list = self.load_images(source_path)

    def load_images(self, source_path: str):
        gt_folder = os.path.join(source_path, "DIV2K_train_HR")
        lr_folder = os.path.join(source_path, "DIV2K_train_LR_bicubic", "X2")

        gt = []
        for file in tqdm(
            sorted(glob.glob(os.path.join(gt_folder, "*.png"))),
            desc="Groundtruth loading",
            unit="Files",
        ):
            image = Image.open(file).convert("L")
            image = to_tensor(image)
            gt.append(image)

        lr = []
        for file in tqdm(
            sorted(glob.glob(os.path.join(lr_folder, "*.png"))),
            desc="LR loading",
            unit="Files",
        ):
            image = Image.open(file).convert("L")
            image = to_tensor(image)
            lr.append(image)

        return gt, lr

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, index: int):
        gt = self.gt_list[index]
        lr = self.lr_list[index]

        return lr, gt
