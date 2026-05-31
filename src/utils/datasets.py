import glob
import os
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from torchvision.transforms.functional import crop

class SimDataset(torch.utils.data.Dataset):
    def __init__(self, source_path: str):
        is_folder = os.path.isdir(source_path)

        self.data = []
        if is_folder:
            for image_path in sorted(glob.glob(os.path.join(source_path, "*.png"))):
                image = Image.open(image_path)
                self.data.append(image)
        else:
            image = Image.open(source_path)
            self.data.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = pil_to_tensor(image)
        return image


class Div2k(torch.utils.data.Dataset):
    def __init__(self, source_path: str, target_size = 512, upsampling = 2):
        super().__init__()
        self.target_size = target_size
        self.input_size = target_size // upsampling

        self.gt_list, self.lr_list = self.load_images(source_path)
    
    def load_images(self, source_path: str):
        gt_folder = os.path.join(source_path, "DIV2K_train_HR")
        lr_folder = os.path.join(source_path, "DIV2K_train_LR_bicubic", "X2")

        gt = []
        for file in tqdm(sorted(glob.glob(os.path.join(gt_folder, "*.png"))), desc="Groundtruth loading", unit="Files"):
            image = Image.open(file).convert('L')
            image = to_tensor(image)
            gt.append(image)
        
        lr = []
        for file in tqdm(sorted(glob.glob(os.path.join(lr_folder, "*.png"))), desc="LR loading", unit="Files"):
            image = Image.open(file).convert('L')
            image = to_tensor(image)
            lr.append(image)
        
        return gt, lr

    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, index: int):
        gt = self.gt_list[index]
        lr = self.lr_list[index]
        
        top_lr = torch.randint(0, lr.shape[1] - self.input_size + 1, (1,)).item()
        left_lr = torch.randint(0, lr.shape[2] - self.input_size + 1, (1,)).item()
        top_gt = top_lr * 2
        left_gt = left_lr * 2
        
        gt = crop(gt, int(top_gt), int(left_gt), self.target_size, self.target_size)
        lr = crop(lr, int(top_lr), int(left_lr), self.input_size, self.input_size)
        
        return lr, gt


def get_data(source_path: str, batch_size: int = 1):
    dataset = SimDataset(source_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4
    )
    return dataset, dataloader
