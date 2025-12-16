import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO

# =========================
# 路径配置
# =========================
coco_root = "/pub/data/lz/autoComposition/autocomp_master/dataset/COCO"
image_dir = os.path.join(coco_root, "val2017")
ann_file = os.path.join(coco_root, "annotations/instances_val2017.json")
occluded_ann_file = os.path.join(coco_root, "annotations/occluded_coco.pkl")

# 输出路径
output_root = "/pub/data/lz/autoComposition/autocomp_master/dataset/forComp_occluded_coco/"
output_foreground = os.path.join(output_root, "foreground")
output_vismask = os.path.join(output_root, "objmask_visiable")
output_amodal = os.path.join(output_root, "objmask_amodal")
os.makedirs(output_foreground, exist_ok=True)
os.makedirs(output_vismask, exist_ok=True)
os.makedirs(output_amodal, exist_ok=True)

# 日志文件
foreground_log = open(os.path.join(output_root, "foreground.txt"), "w")
vismask_log = open(os.path.join(output_root, "vismask.txt"), "w")
amodalmask_log = open(os.path.join(output_root, "amodalmask.txt"), "w")
occluded_log = open(os.path.join(output_root, "occluded.txt"), "w")

# =========================
# 加载 COCO 标注和 Occluded COCO
# =========================
coco = COCO(ann_file)

with open(occluded_ann_file, "rb") as f:
    occluded_data = pickle.load(f)

# for i in range(1000):
#     print(occluded_data[i])


# 建立快速查询被遮挡物体的字典: key = filename_instid, value = (vis_mask, amodal_mask)
occluded_dict = {}
for entry in occluded_data:
    filename, category, inst_id, bbox, rubish = entry
    key = f"{filename}_{inst_id}"
    occluded_dict[key] = 1

# =========================
# 遍历所有 COCO 图片
# =========================
count = 0
for img_id in tqdm(coco.getImgIds(), desc="Processing COCO"):    
    count += 1
    if count > 5:
        break
    img_info = coco.loadImgs([img_id])[0]
    width, height = img_info["width"], img_info["height"]
    img_path = os.path.join(image_dir, img_info["file_name"])

    # if img_info['file_name'] != "000000389933.jpg":
    #     continue

    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("RGBA")
    rgba = np.array(image, dtype=np.uint8)

    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

    for ann_idx,ann in enumerate(anns):
        if ann.get("iscrowd", 0):
            continue

        obj_id = ann["id"]
        key = f"{img_info['file_name']}_{ann_idx}"

        # 判断是否在 occluded_dict
        if key in occluded_dict:
            is_occluded = True
        else:
            is_occluded = False

        # 没有遮挡，vis/amodal 相同，用 segmentation 生成 mask
        segmentation = ann["segmentation"]
        rle = maskUtils.frPyObjects(segmentation, height, width)
        mask = maskUtils.decode(rle)
        mask = mask.max(axis=2) if mask.ndim == 3 else mask
        vis_mask = (mask > 0).astype(np.uint8) * 255
        amodal_mask = vis_mask.copy()

        base_name = f"{img_info['file_name'].replace('.jpg','')}_{ann_idx}_{count}"

        # 保存可见 mask
        vis_img = Image.fromarray(vis_mask)
        vis_img.save(os.path.join(output_vismask, base_name + ".png"))
        vismask_log.write(base_name + ".png\n")

            # # 保存 amodal mask
            # amodal_img = Image.fromarray(amodal_mask)
            # amodal_img.save(os.path.join(output_amodal, base_name + ".png"))
            # amodalmask_log.write(base_name + ".png\n")
            

        if is_occluded:
            occluded_log.write(base_name + f".png   {ann_idx}\n")
        else:
            # 保存前景
            mask_bool = vis_mask.astype(bool)
            fg_rgba = rgba.copy()
            fg_rgba[~mask_bool] = 0
            fg_img = Image.fromarray(fg_rgba, mode="RGBA")
            fg_img.save(os.path.join(output_foreground, base_name + ".png"))
            foreground_log.write(base_name + f".png   {ann_idx}\n")

# =========================
# 关闭日志
# =========================
foreground_log.close()
vismask_log.close()
amodalmask_log.close()
occluded_log.close()

print("✅ Occluded COCO 数据集处理完成！")
