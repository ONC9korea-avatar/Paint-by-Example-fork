import os
import os.path as osp

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

import torchvision.transforms as T
import albumentations as A

transform_ref_image = A.Compose([
    A.Resize(height=224, width=224), # for CLIP
    # A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20),
    # A.Blur(p=0.3),
    # A.ElasticTransform(p=0.3),
])

transform_random_hflip = A.Compose(
    [A.HorizontalFlip(p=0.5)],
    additional_targets={
        'inpaint_image': 'image',
        'inpaint_mask': 'mask',
        'ref_imgs': 'image',
        'densepose': 'mask'
    }
)

transform_color = A.Compose(
    [
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=0.5)
    ],
    additional_targets={
        'GT': 'image',
        'ref_imgs': 'image',
        'inpaint_image': 'image',
    }
)

transform_pose = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5, border_mode=cv.BORDER_CONSTANT, value=(0,0,0)),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=0.5, border_mode=cv.BORDER_CONSTANT, value=(0,0,0))],
    additional_targets={
        # group 1
        'GT': 'image',
        'inpaint_image': 'image',
        'inpaint_mask': 'mask',
        'densepose': 'mask'
    }
)

def to_tensor(image, normalize=True, clip=False):
    transforms = [T.ToTensor()]

    if normalize:
        if clip:
            mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        else:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        transforms.append(T.Normalize(mean, std))

    return T.Compose(transforms)(image)

class VITONHDDataset_PBE(Dataset):
    def __init__(
            self,
            state,
            dataset_dir, 
            img_height, 
            img_width,
            is_sorted = False,       
            **kwargs
        ):
        self.state = state # train, validation, or test

        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width        
        
        self.is_sorted = is_sorted
        self.kwargs = kwargs

        self.names = os.listdir(osp.join(self.dataset_dir, 'cloth'))
        if is_sorted:
            self.names.sort()


    def __len__(self):
        return len(self.names)
    
    def get_image_path(self, dir, idx):
        if dir == 'agnostic-mask':
            return osp.join(self.dataset_dir, dir, self.names[idx].replace('.jpg', '_mask.png'))
        else:
            return osp.join(self.dataset_dir, dir, self.names[idx])
    
    def __getitem__(self, idx):
        item = dict(
            image = cv.imread(self.get_image_path("gt_cloth_warped", idx)),
            
            inpaint_image = cv.imread(self.get_image_path("gt_cloth_warped+agn_mask", idx)),
            inpaint_mask = cv.imread(self.get_image_path("agnostic-mask", idx)),

            ref_imgs = cv.imread(self.get_image_path("cloth", idx)),

            densepose = cv.imread(self.get_image_path('image-densepose', idx)),
        )

        # resize all images/masks
        # convert to rgb
        for k in item.keys():
            if k == 'ref_imgs':
                item[k] = transform_ref_image(image=item[k])['image']
            else:
                item[k] = cv.resize(item[k], (self.img_width, self.img_height))
            
            item[k] = cv.cvtColor(item[k], cv.COLOR_BGR2RGB)
            
            if k == 'inpaint_mask' or k == 'densepose':
                item[k] = item[k][:,:,0] # to one-chennel

        # apply augmentation
        if self.state == "train": 
            item.update(**transform_random_hflip(**item))

            transform_color_keys = ['image', 'ref_imgs', 'inpaint_image']
            item.update(**transform_color(**{k:item[k] for k in transform_color_keys}))

            transform_pose_keys = ['image', 'inpaint_image', 'inpaint_mask', 'densepose']
            item.update(**transform_pose(**{k:item[k] for k in transform_pose_keys}))

        # make inpaint image
        mask_float = item['inpaint_mask'][:,:,None].astype(np.float32) / 255.
        item['inpaint_image'] = item['inpaint_image'] * (1-mask_float) + 128 * mask_float

        # to_tensor
        for k in item.keys():
            if k == 'ref_imgs':
                item[k] = to_tensor(item[k], clip=True)
            elif k == 'inpaint_mask' or k=='densepose':
                item[k] = to_tensor(item[k], normalize=False)
            else:
                item[k] = to_tensor(item[k])
        
        # rename key 'image' -> 'GT'
        item['GT'] = item.pop('image')

        return item

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = VITONHDDataset_PBE(
        'train',
        '/home/jun/Paint-by-Example-fork/dataset/VITON-HD/train',
        512, 384, True
        )
    n = 10
    data_list = [dataset[i] for i in range(n)]
    
    fig, axs = plt.subplots(n, len(data_list[0].keys()), figsize=(10,20), dpi=300)

    keys = ['GT', 'ref_imgs', 'inpaint_image', 'inpaint_mask', 'densepose']
    for i, (ax, data) in enumerate(zip(axs, data_list)):
        for a, k in zip(ax, keys):
            if i == 0:
                a.set_title(k)

            im = data[k].permute(1, 2, 0)
            # for s in a.spines:
            #     a.spines[s].set_visible(False)
            a.set_xticks([])
            a.set_yticks([])

            if k == 'inpaint_mask' or k == 'densepose':
                a.imshow(im, cmap='gray')
            else:
                a.imshow(im)

    fig.savefig('vitonhd_dataset.png')
    
    