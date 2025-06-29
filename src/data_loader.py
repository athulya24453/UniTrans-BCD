import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import rasterio
import cv2

class RSCDataset(Dataset):
    """
    Remote Sensing Change Detection Dataset.
    Works with any folder tree: 
      root_dir/
        ├── t1/
        ├── t2/
        └── gt/      all containing {train,val,test} subfolders of .npy or .tif/.png
    """
    def __init__(self, root_dir, split='train', patch_size=None, transforms=None):
        """
        Args:
            root_dir (str): e.g. '/storage2/.../sysu_preprocessed'
            split (str): one of ['train','val','test']
            patch_size (int or None): if None, no cropping (we assume pre-tiling done)
            transforms: albumentations Compose
        """
        self.transforms = transforms
        # build file lists
        self.t1_paths = sorted(self._gather(os.path.join(root_dir, 't1', split)))
        self.t2_paths = sorted(self._gather(os.path.join(root_dir, 't2', split)))
        self.gt_paths = sorted(self._gather(os.path.join(root_dir, 'gt', split)))
        assert len(self.t1_paths)==len(self.t2_paths)==len(self.gt_paths), \
            f"Mismatch lengths: {len(self.t1_paths)}, {len(self.t2_paths)}, {len(self.gt_paths)}"

    def _gather(self, folder):
        """Collect .npy, .tif or .png files in folder."""
        exts = ('.npy','.tif','.tiff','.png')
        return [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)]

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, idx):
        p1, p2, pg = self.t1_paths[idx], self.t2_paths[idx], self.gt_paths[idx]

        # load img1
        if p1.lower().endswith('.npy'):
            img1 = np.load(p1)
            img2 = np.load(p2)
            mask = np.load(pg)
        else:
            # tif or png—use rasterio for tifs, cv2 for png
            if p1.lower().endswith(('.tif','.tiff')):
                with rasterio.open(p1) as src1, rasterio.open(p2) as src2:
                    img1 = src1.read().transpose(1,2,0)
                    img2 = src2.read().transpose(1,2,0)
                with rasterio.open(pg) as srcg:
                    mask = srcg.read(1)
            else:
                # png
                import cv2
                img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(pg, cv2.IMREAD_GRAYSCALE)

        # apply transforms if requested (e.g. random crop)
        if self.transforms:
            augmented = self.transforms(image=img1, image0=img2, mask=mask)
            img1, img2, mask = augmented['image'], augmented['image0'], augmented['mask']

        # convert to CHW if needed downstream, but leave as HWC here
        return img1, img2, mask

def get_dataloader(root_dir,
                   split='train',
                   batch_size=8,
                   patch_size=None,
                   overlap=0,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=True):
    """
    Returns a DataLoader for RSCDataset.
    If patch_size is provided, applies RandomCrop to that size.
    """
    transforms = None
    if patch_size:
        transforms = A.Compose([
            A.RandomCrop(width=patch_size, height=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'image0':'image','mask':'mask'})

    dataset = RSCDataset(root_dir, split, patch_size, transforms)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)
