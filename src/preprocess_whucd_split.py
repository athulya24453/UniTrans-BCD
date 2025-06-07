import os
import numpy as np
import rasterio
from tqdm import tqdm

def convert_whucd_split_to_numpy(
    src_root='/storage2/ChangeDetection/datasets/WHU-CD',
    dest_root='/storage2/ChangeDetection/datasets/whu_cd_preprocessed'
):
    # Source directories
    t1_dir = os.path.join(src_root, 'image_data', '2012', 'splitted_images', 'train', 'image')
    t2_dir = os.path.join(src_root, 'image_data', '2016', 'splitted_images', 'train', 'image')
    gt_dir = os.path.join(src_root, 'image_data', 'change_label', 'splitted_images', 'train', 'label')

    # Output directories
    out_t1 = os.path.join(dest_root, 't1', 'train')
    out_t2 = os.path.join(dest_root, 't2', 'train')
    out_gt = os.path.join(dest_root, 'gt', 'train')
    os.makedirs(out_t1, exist_ok=True)
    os.makedirs(out_t2, exist_ok=True)
    os.makedirs(out_gt, exist_ok=True)

    # Process files
    filenames = sorted(os.listdir(t1_dir))
    for fname in tqdm(filenames, desc="Converting WHU-CD to .npy"):
        if not fname.endswith('.tif'):
            continue
        t1_path = os.path.join(t1_dir, fname)
        t2_path = os.path.join(t2_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        with rasterio.open(t1_path) as src:
            t1 = src.read().transpose(1, 2, 0)
        with rasterio.open(t2_path) as src:
            t2 = src.read().transpose(1, 2, 0)
        with rasterio.open(gt_path) as src:
            label = src.read(1)

        # Save .npy
        np.save(os.path.join(out_t1, fname.replace('.tif', '.npy')), t1)
        np.save(os.path.join(out_t2, fname.replace('.tif', '.npy')), t2)
        np.save(os.path.join(out_gt, fname.replace('.tif', '.npy')), label)

    print(f"Finished converting and saving {len(filenames)} patch triplets to:")
    print(f"   {dest_root}")

if __name__ == "__main__":
    convert_whucd_split_to_numpy()
