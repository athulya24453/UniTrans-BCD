import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import importlib.util

# Dynamically load data_loader.py
loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/data_loader.py'))
spec = importlib.util.spec_from_file_location("data_loader", loader_path)
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)
get_dataloader = data_loader.get_dataloader

def main():
    sysu_root = '/storage2/ChangeDetection/datasets/sysu_preprocessed'

    loader = get_dataloader(
        root_dir=sysu_root,
        split='train',
        batch_size=4,
        patch_size=None  # already tiled
    )

    imgs1, imgs2, masks = next(iter(loader))

    print("Time1 batch shape:", imgs1.shape)
    print("Time2 batch shape:", imgs2.shape)
    print("Mask  batch shape:", masks.shape)

    # Optional: visualize first sample
    img_a = imgs1[0].cpu().numpy().astype(np.uint8)
    img_b = imgs2[0].cpu().numpy().astype(np.uint8)
    mask  = masks[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_a)
    axes[0].set_title('Time 1')
    axes[1].imshow(img_b)
    axes[1].set_title('Time 2')
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Label')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
