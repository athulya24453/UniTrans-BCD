# src/sysu_mae_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SysuMAEDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to t1/train/ directory containing .npy image patches.
        """
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        image = np.load(file_path)  # shape: (256, 256, 3), dtype: uint8 or float32

        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # shape: (3, 256, 256)

        return image
