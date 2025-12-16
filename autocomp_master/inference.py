import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr

import numpy as np
from dataset import OcclusionSegmentationDataset
from unet import UNet
from swin_unet import SwinUNet

from pathlib import Path
import cv2

from segment_anything import SamPredictor, sam_model_registry

import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# -----------------------
# 加载模型
# -----------------------

def get_available_models(ckpt_dir="/pub/data/lz/autoComposition/autocomp_master/ckpt/"):
    """获取指定目录下的所有可用模型文件"""
    ckpt_dir = Path(ckpt_dir)
    return sorted([str(p) for p in ckpt_dir.glob("*.pth")])

def load_model(ckpt_path="", device="cuda" if torch.cuda.is_available() else "cpu"):
    model = SwinUNet()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# model = load_model()

def segment_fg(method=None,fg_img=None):  #进来的都是RGBA格式
    fg_img_array = np.array(fg_img)
    red, green, blue, alpha = fg_img_array.T

    if method == "FromWhite":
        # if np.array(fg_img)[:,:,3][0,0]!=0:  #第一个元素透明度不是零，RGB来的，pix2gestalt生成来的，用白色扣图一下
        
        # 定义白色范围 (R, G, B) = (255, 255, 255)
        # 由于可能存在非纯白光，设置一个容差范围
        white_threshold = 240  # 容差值，可根据实际情况调整
        white_pixels = (red.T > white_threshold) & (green.T > white_threshold) & (blue.T > white_threshold)
        
        # 将白色像素的alpha通道设为0（透明）
        fg_img_array[..., 3][white_pixels] = 0
        
        # 将处理后的数组转换回Image对象
        fg_img = Image.fromarray(fg_img_array)

        fg_amodal_mask = np.where(np.array(fg_img)[:,:,3]==0,0,1)
        
    elif method == "Grabcut":
        # 转换为BGR格式（cv2要求）
        bgr = cv2.cvtColor(np.array(fg_img)[..., :3], cv2.COLOR_RGB2BGR)
        mask = np.zeros(bgr.shape[:2], np.uint8)

        # 使用整图作为初始矩形
        height, width = bgr.shape[:2]
        rect = (1, 1, width-2, height-2)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # 运行GrabCut
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # 转换mask为二值（0背景，1前景）
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        fg_amodal_mask = mask2

        
    elif method == "SAM":
        sam = sam_model_registry["vit_h"](checkpoint="/pub/data/lz/autoComposition/autocomp_master/libraries/sam_vit_h_4b8939.pth") #初始化
        # RGB图像输入
        rgb_image = np.array(fg_img)[..., :3]

        predictor = SamPredictor(sam)
        predictor.set_image(rgb_image)

        # 整图居中提示框
        h, w = rgb_image.shape[:2]
        input_box = np.array([[0, 0, w - 1, h - 1]])  # x0, y0, x1, y1

        masks, scores, logits = predictor.predict(
            box=input_box,
            multimask_output=False
        )

        fg_amodal_mask = 1-masks[0].astype(np.uint8)  #这个是反的

    else:
        print("没有这种方法傻逼，写nm错了")

    return fg_amodal_mask


    


# -----------------------
# 推理函数
# -----------------------

def preprocess(image, size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(image)

def inference(fg_img, bg_img, model_path):
    global model
    global current_model_path
    if model_path != current_model_path:  # 当前加载的模型路径
        model = load_model(model_path)
        current_model_path = model_path  # 更新当前加载的模型路径

    device = next(model.parameters()).device

    image_W,image_H = bg_img.size

    # 预处理图像
    fg_tensor = preprocess(fg_img.convert('RGB')).to(device)
    bg_tensor = preprocess(bg_img).to(device)

    # 拼接输入
    input_tensor = torch.cat([fg_tensor, bg_tensor], dim=0).unsqueeze(0)  # [1, 6, H, W]

    # 前向推理
    with torch.no_grad():
        pred_mask = model(input_tensor)[0, 0]  # [H, W]
        mask = (pred_mask > 0.5).float()  # 二值化

    # 合成图
    fg_np = fg_tensor.permute(1, 2, 0).cpu().numpy()
    bg_np = bg_tensor.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    mask_from_net = mask_np * 1

    fg_amodal_mask = segment_fg(method= config['inference_seg'],fg_img=fg_img)   #得到fg_amodal_mask  FromWhite Grabcut SAM

    mask_np = mask_np *  cv2.resize(fg_amodal_mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)    #求交得到最终合成所用mask,最近邻resize
    # mask_np = mask_np *  cv2.resize(fg_amodal_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)    #求交得到最终合成所用mask,最近邻resize

    composite = fg_np * mask_np[..., None] + bg_np * (1 - mask_np[..., None])
    composite_img = (composite * 255).astype(np.uint8)
    mask_img = (mask_np * 255).astype(np.uint8)

    
    # print(fg_img.mode)

    #无遮挡合成图,
    fg_amodal_mask_3d = np.stack([fg_amodal_mask]*3, axis=2)
    no_occ_composite = np.array(fg_img.convert('RGB'))*fg_amodal_mask_3d + np.array(bg_img)*(1-fg_amodal_mask_3d) #可能不一般大
    no_occ_composite_img = no_occ_composite.astype(np.uint8)
    
    
    mask_from_net_img = (mask_from_net * 255).astype(np.uint8)

    mask_output = Image.fromarray(mask_img).resize((image_W,image_H))   #其实前景mask不应该恢复成背景大小
    mask_from_net_img_output = Image.fromarray(mask_from_net_img).resize((image_W,image_H))
    no_occ_composite_img_output = Image.fromarray(no_occ_composite_img).resize((image_W,image_H))
    composite_img_output = Image.fromarray(composite_img).resize((image_W,image_H))

    return mask_output, mask_from_net_img_output, no_occ_composite_img_output, composite_img_output

# -----------------------
# Gradio 接口
# -----------------------

def run_web_ui():
    models = get_available_models()
    iface = gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(image_mode='RGBA',type="pil", label="前景图"),
            gr.Image(type="pil", label="背景图"),
            gr.Dropdown(
                choices=models,
                value=models[0],
                label="选择模型"
            )
        ],
        outputs=[
            gr.Image(type="pil", label="前景求交后 Mask"),
            gr.Image(type="pil", label="预测 Mask"),
            gr.Image(type="pil", label="不考虑遮挡合成图"),
            gr.Image(type="pil", label="合成图")
        ],
        title="遮挡恢复合成 Demo",
        description="上传前景图（带遮挡）和背景图，自动恢复前景 mask 并完成图像合成"
    )
    iface.launch()

# -----------------------
# 入口函数
# -----------------------

if __name__ == "__main__":
    # 修改：初始加载第一个可用模型
    available_models = get_available_models()
    if not available_models:
        raise FileNotFoundError("未在指定目录下找到模型文件")
    
     
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    first_model = "/pub/data/lz/autoComposition/autocomp_master/ckpt/SwinUnet_final_table.pth"
        
    model = load_model(first_model)
    current_model_path = first_model

    run_web_ui()
