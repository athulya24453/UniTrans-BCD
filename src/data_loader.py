import os
import rasterio
import numpy as np

def tile_whu_cd(root_dir, patch_size=256, overlap=0):
    src_dir = os.path.join(root_dir, 'source', 'whu_cd')
    out_dir = os.path.join(root_dir, 'processed', 'whu_cd')
    for split in ['t1','t2','gt']:
        os.makedirs(os.path.join(out_dir, split, 'train'), exist_ok=True)

    # Load full images
    img12 = rasterio.open(os.path.join(src_dir, 'whu_cd_2012.tif')).read().transpose(1,2,0)
    img16 = rasterio.open(os.path.join(src_dir, 'whu_cd_2016.tif')).read().transpose(1,2,0)
    gt   = rasterio.open(os.path.join(src_dir, 'whu_cd_gt.tif')).read(1)

    h, w, _ = img12.shape
    stride = patch_size - overlap
    idx = 0
    for y in range(0, h-patch_size+1, stride):
        for x in range(0, w-patch_size+1, stride):
            p1 = img12[y:y+patch_size, x:x+patch_size]
            p2 = img16[y:y+patch_size, x:x+patch_size]
            m  = gt[y:y+patch_size, x:x+patch_size]

            # Save as NumPy arrays
            np.save(os.path.join(out_dir, 't1', 'train', f'{idx:05d}.npy'), p1)
            np.save(os.path.join(out_dir, 't2', 'train', f'{idx:05d}.npy'), p2)
            np.save(os.path.join(out_dir, 'gt', 'train', f'{idx:05d}.npy'), m)
            idx += 1

    print(f'Tiled {idx} patches from WHU-CD.')
