#根据visable mask的txt中找到foreground文件夹中没有的，进行生成，记得用pix2gestalt环境


import sys
sys.path.append("/pub/data/lz/autoComposition/generate_dataset/pix2gestalt-main/pix2gestalt")
import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from inference import load_model_from_config, run_inference

# 模型配置
config_path = '/pub/data/lz/autoComposition/generate_dataset/pix2gestalt-main/pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml'
ckpt_path = '/pub/data/lz/autoComposition/generate_dataset/pix2gestalt-main/pix2gestalt/ckpt/epoch=000005.ckpt'
# device = 'cpu'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
config = OmegaConf.load(config_path)
model = load_model_from_config(config, ckpt_path, device)
model.eval()

# 路径配置
filefold = '/pub/data/lz/autoComposition/autocomp_master/dataset/forComp_COCOA_occlude_filter/'
foreground_dir = filefold + 'foreground'
vismask_txt_path =  filefold + 'vismask.txt'
foreground_txt_path =  filefold + 'foreground.txt'
vismask_dir =  filefold + 'objmask_visiable'

# original_image_dir = '/pub/data/lz/autoComposition/autocomp_master/dataset/BSDS/images/train/data'
original_image_dir = '/pub/data/lz/autoComposition/autocomp_master/dataset/COCO2014/train2014/'



foreground_from_generate_path =  filefold + 'foreground_from_generate.txt'

#/pub/data/lz/autoComposition/autocomp_master/dataset/COCO2014/train2014/COCO_train2014_000000000009.jpg

# 推理参数
guidance_scale = 2.0
n_samples = 1
ddim_steps = 50
# ddim_steps = 2

# 读取数据
with open(vismask_txt_path, 'r') as f:
    vismask_list = [line.strip() for line in f.readlines()]
with open(foreground_txt_path, 'r') as f:
    foreground_list = set(line.strip() for line in f.readlines())
    # foreground_list = [line.strip() for line in f.readlines()]

# 统计需要推理的项目
todo_list = [filename for filename in vismask_list if filename not in foreground_list]  
#这方法难说不垃圾，是逐个遍历看前景中有没有的，其实可以读取出vismask.txt和foreground.txt作为两个列表，同时从头开始遍历，如果名字相同，则两个列表同时取下一个；如果不相同，则为该物体执行一次inference得到完整物体

# 进度条开始
i = 0
for filename in tqdm(todo_list, desc="生成完整物体", unit="obj"):
    i = i + 1
    # if i < 1154:
    #     continue
    base_name = os.path.splitext(filename)[0]  # e.g. 100075_3
    if 'COCO2014' in original_image_dir:
        image_id = base_name.split("_")[0]+"_"+base_name.split("_")[1]+"_"+base_name.split("_")[2]
    else:
        image_id = base_name.split("_")[0]

    image_path = os.path.join(original_image_dir, f"{image_id}.jpg")
    mask_path = os.path.join(vismask_dir, filename)

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"跳过缺失文件: {image_path}")
        continue

    input_img = Image.open(image_path).convert("RGB")
    width, height = input_img.size
    # wbn_help_size = 64
    resize_image = np.array(input_img.resize((256, 256)))
    visible_mask = np.array(Image.open(mask_path).convert("L").resize((256, 256)))
    # resize_image = np.array(input_img.resize((wbn_help_size,wbn_help_size)))
    # visible_mask = np.array(Image.open(mask_path).convert("L").resize((wbn_help_size,wbn_help_size)))
    with torch.no_grad():
        output_images = run_inference(resize_image, visible_mask, model, guidance_scale, n_samples, ddim_steps)

    out_img = output_images[0]
    result_img = Image.fromarray(out_img).resize((width, height))
    save_path = os.path.join(foreground_dir, filename)
    result_img.save(save_path)

    with open(foreground_from_generate_path, 'a') as fg_log:
        fg_log.write(filename + "\n")

print("所有缺失物体已完成推理！")
