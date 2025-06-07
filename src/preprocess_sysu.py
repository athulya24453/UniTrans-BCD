import os
import numpy as np
import cv2
from tqdm import tqdm

def convert_sysu_png_to_numpy(
    src_root='/storage2/ChangeDetection/datasets/SYSU',
    dest_root='/storage2/ChangeDetection/datasets/sysu_preprocessed'
):
    splits = ['train', 'val', 'test']
    for split in splits:
        t1_dir = os.path.join(src_root, split, 'time1')
        t2_dir = os.path.join(src_root, split, 'time2')
        gt_dir = os.path.join(src_root, split, 'label')

        out_t1 = os.path.join(dest_root, 't1', split)
        out_t2 = os.path.join(dest_root, 't2', split)
        out_gt = os.path.join(dest_root, 'gt', split)
        os.makedirs(out_t1, exist_ok=True)
        os.makedirs(out_t2, exist_ok=True)
        os.makedirs(out_gt, exist_ok=True)

        filenames = sorted(os.listdir(t1_dir))
        for fname in tqdm(filenames, desc=f"Processing {split}"):
            if not fname.endswith('.png'):
                continue

            t1_path = os.path.join(t1_dir, fname)
            t2_path = os.path.join(t2_dir, fname)
            gt_path = os.path.join(gt_dir, fname)

            # Load color images as RGB
            t1 = cv2.cvtColor(cv2.imread(t1_path), cv2.COLOR_BGR2RGB)
            t2 = cv2.cvtColor(cv2.imread(t2_path), cv2.COLOR_BGR2RGB)

            # Load label as grayscale
            label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # Save as .npy
            base = fname.replace('.png', '.npy')
            np.save(os.path.join(out_t1, base), t1)
            np.save(os.path.join(out_t2, base), t2)
            np.save(os.path.join(out_gt, base), label)

        print(f"✅ Finished saving {len(filenames)} patch triplets to: {split}/")

if __name__ == "__main__":
    convert_sysu_png_to_numpy()
