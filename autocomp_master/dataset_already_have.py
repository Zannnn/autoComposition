import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2

class AlreadyHaveBackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 结构如下：
        └── root_dir/
            ├── images/
            ├    ├── back/
            ├    └──fore/
            └── annotation/
                ├── back/
                └──fore/

        """
        # root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_basket_bottle"
        self.background_dir = os.path.join(root_dir,"images/back")
        self.foreground_dir = os.path.join(root_dir, "images/fore")
        self.mask_dir = os.path.join(root_dir, "annotation/fore")
        # self.amodalmask_dir = os.path.join(root_dir, "objmask_amodal")
        self.image_names = sorted(os.listdir(self.foreground_dir))
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        # fg = Image.open(os.path.join(self.foreground_dir, name)).convert("RGB")


        #-------直接得到foreground是RGBA，来自pix2gestalt的是RGB，我得得到fg_mask-----------------------------------------------------------------
        fg = Image.open(os.path.join(self.foreground_dir, name))
        # # print(name)
        # if fg.mode == 'RGBA': #来自json
        fg_array = np.array(fg)
        is_white = (fg_array[:, :, :3] >= 240).all(axis=2)
        fg_mask = Image.fromarray((np.where(is_white, 0, 1) * 255).astype(np.uint8))
        # fg = fg.convert("RGB")
        # else :                 #来自生成
        #     fg_mask = Image.open(os.path.join(self.amodalmask_dir, name)).convert("L")   #好像不对啊，本来就是生成的，哪来的mask
        
        
        #--------------------------------------------------------------------------

        bg = Image.open(os.path.join(self.background_dir,   name.split('_')[0] )).convert("RGB")
        target = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        
        fg_mask = self.transform(fg_mask)
        fg_mask = (fg_mask > 0.5).float()     #进行了二值化

        bg = self.transform(bg)
        fg = self.transform(fg)
        target = self.transform(target)
        target = (target > 0.5).float()     #进行了二值化

        # Concatenate background, foreground, and mask as input
        input_tensor = torch.cat([fg, bg], dim=0)
        # mask = mask.permute(0,2,1)
        # print(input_tensor.shape)
        # print(mask.shape)
        return input_tensor, fg_mask, target  # GT mask here is the ground truth visible mask
