#voc这数据集不知道谁被挡，因此不知道谁用pix2gestalt，因此未完成。
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import xml.etree.ElementTree as ET

# =========================
# 路径配置（按需修改）
# =========================
voc_root = "/pub/data/lz/autoComposition/autocomp_master/dataset/VOC2012"  # VOC2012 根目录（包含 JPEGImages / Annotations / SegmentationObject / SegmentationClass / ImageSets 等）
image_dir = os.path.join(voc_root, "JPEGImages")
ann_xml_dir = os.path.join(voc_root, "Annotations")
segobj_dir = os.path.join(voc_root, "SegmentationObject")   # 若无此目录，脚本会自动回退到 bbox
# 可选：仅处理某个列表里的图片（如 train/val/trainval）。若为 None，则处理 JPEGImages 下所有图片
imageset_txt = os.path.join(voc_root, "ImageSets", "Main", "trainval.txt")  # 或 "train.txt"/"val.txt"
# 输出目录（与 BSDS 示例一致的结构名）
output_root = "/pub/data/lz/autoComposition/autocomp_master/dataset/forComp_dataset_voc2012/"
output_foreground = os.path.join(output_root, "foreground")
output_vismask = os.path.join(output_root, "objmask_visiable")
# =========================

os.makedirs(output_foreground, exist_ok=True)
os.makedirs(output_vismask, exist_ok=True)

# 打开日志文件
foreground_log = open(os.path.join(output_root, "foreground.txt"), "w")
vismask_log = open(os.path.join(output_root, "vismask.txt"), "w")

# 读取待处理图片列表
if imageset_txt is not None and os.path.isfile(imageset_txt):
    with open(imageset_txt, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
else:
    # 扫描 JPEGImages 下的所有图片名（去掉扩展名）
    ids = []
    for fn in os.listdir(image_dir):
        name, ext = os.path.splitext(fn)
        if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            ids.append(name)
    ids.sort()

def load_image(image_path):
    # 统一读为 RGBA，便于透明背景
    img = Image.open(image_path).convert("RGBA")
    return img

def save_mask_and_foreground(image_rgba: Image.Image, mask_bool: np.ndarray, base_name: str, count: int):
    """
    按给定布尔 mask 保存：
    - 可见 mask（二值 0/255）到 objmask_visiable/
    - foreground（透明背景 RGBA）到 foreground/
    并记录日志
    """
    h, w = mask_bool.shape
    # 保存 vis mask
    vis_mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    vis_name = f"{base_name}_{count}.png"
    vis_path = os.path.join(output_vismask, vis_name)
    vis_mask_img.save(vis_path)
    vismask_log.write(vis_name + "\n")

    # 保存 foreground
    rgba = np.array(image_rgba, dtype=np.uint8)
    rgba[~mask_bool] = 0
    fg_img = Image.fromarray(rgba, mode="RGBA")
    fg_name = f"{base_name}_{count}.png"
    fg_path = os.path.join(output_foreground, fg_name)
    fg_img.save(fg_path)
    foreground_log.write(fg_name + "\n")

# def get_bbox_masks_from_xml(xml_path, width, height):
#     """
#     从 VOC 的 XML 中读取 bboxes，生成矩形 mask 列表（每个对象一个布尔 mask）
#     返回：[(obj_idx, mask_bool), ...]
#     """
#     masks = []
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     objects = root.findall("object")
#     for idx, obj in enumerate(objects, start=1):
#         bnd = obj.find("bndbox")
#         if bnd is None:
#             continue
#         try:
#             xmin = int(float(bnd.find("xmin").text))
#             ymin = int(float(bnd.find("ymin").text))
#             xmax = int(float(bnd.find("xmax").text))
#             ymax = int(float(bnd.find("ymax").text))
#         except Exception:
#             continue
#         # clamp 到图像范围
#         xmin = max(0, min(xmin, width - 1))
#         ymin = max(0, min(ymin, height - 1))
#         xmax = max(0, min(xmax, width))
#         ymax = max(0, min(ymax, height))
#         if xmax <= xmin or ymax <= ymin:
#             continue

#         mask = Image.new("L", (width, height), 0)
#         draw = ImageDraw.Draw(mask)
#         draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
#         mask_bool = np.array(mask, dtype=np.uint8) > 0
#         masks.append((idx, mask_bool))
#     return masks

count = 0  # 全局唯一编号

for img_id in tqdm(ids, desc="Processing VOC2012"):
    if count >5:
        break
    img_path = os.path.join(image_dir, img_id + ".jpg")
    if not os.path.isfile(img_path):
        # 部分数据可能是 .png
        img_path = os.path.join(image_dir, img_id + ".png")
        if not os.path.isfile(img_path):
            # 找不到图像则跳过
            continue

    # 加载图像
    img_rgba = load_image(img_path)
    width, height = img_rgba.size

    processed_any = False

    # 优先使用 SegmentationObject（实例分割）
    seg_path = os.path.join(segobj_dir, img_id + ".png")
    if os.path.isfile(seg_path):
        seg_img = Image.open(seg_path)
        seg_arr = np.array(seg_img, dtype=np.uint8)
        # VOC 约定：0 为背景，255 常作边界/忽略；其他值为实例 ID
        uniq_vals = np.unique(seg_arr)
        instance_vals = [v for v in uniq_vals if v != 0 and v != 255]

        for inst_val in instance_vals:
            mask_bool = (seg_arr == inst_val)
            if not mask_bool.any():
                continue
            count += 1
            base_name = f"{img_id}_{int(inst_val)}"  # 与 BSDS 类似：image_id_实例值
            save_mask_and_foreground(img_rgba, mask_bool, base_name, count)
            processed_any = True

    # 若无实例分割，则回退到 bbox（矩形 mask）
    # if not processed_any:
    #     xml_path = os.path.join(ann_xml_dir, img_id + ".xml")
    #     if os.path.isfile(xml_path):
    #         masks = get_bbox_masks_from_xml(xml_path, width, height)
    #         for obj_idx, mask_bool in masks:
    #             if not mask_bool.any():
    #                 continue
    #             count += 1
    #             base_name = f"{img_id}_{obj_idx}"  # bbox 顺序索引
    #             save_mask_and_foreground(img_rgba, mask_bool, base_name, count)
    #             processed_any = True

    # 若既没有实例分割也没有 bbox，则跳过该图像
    # 也可以打印提示：
    # if not processed_any:
    #     print(f"[WARN] no objects for: {img_id}")

# 关闭日志文件
foreground_log.close()
vismask_log.close()
print("VOC2012 前景与可见 mask 构建完成！")
