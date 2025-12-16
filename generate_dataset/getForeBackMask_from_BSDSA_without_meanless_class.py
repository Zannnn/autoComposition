import json
import os
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils
from tqdm import tqdm

# 路径配置
json_path = "/pub/data/lz/autoComposition/autocomp_master/dataset/BSDSAandCOCOA/annotations/BSDS_amodal_train.json"
image_root = "/pub/data/lz/autoComposition/autocomp_master/dataset/BSDS/images/train/data/"
meanless_file = "/pub/data/lz/autoComposition/autocomp_master/dataset/BSDSAandCOCOA/meanless_class.txt"

# 输出路径
output_filefold = "/pub/data/lz/autoComposition/autocomp_master/dataset/forComp_BSDSA_occlude_filter/"
output_foreground = os.path.join(output_filefold, "foreground")
output_vismask = os.path.join(output_filefold, "objmask_visiable")
output_amodal = os.path.join(output_filefold, "objmask_amodal")

# 创建保存目录
os.makedirs(output_foreground, exist_ok=True)
os.makedirs(output_vismask, exist_ok=True)
os.makedirs(output_amodal, exist_ok=True)

# 打开文件用于写入保存的文件名
foreground_log = open(os.path.join(output_filefold, "foreground.txt"), "w")
vismask_log = open(os.path.join(output_filefold, "vismask.txt"), "w")
amodalmask_log = open(os.path.join(output_filefold, "amodalmask.txt"), "w")

# 加载 meanless 类别
with open(meanless_file, "r", encoding="utf-8") as f:
    meanless_classes = set(line.strip().lower() for line in f if line.strip())

# 加载 JSON 标注文件
with open(json_path, "r") as f:
    data = json.load(f)

# 遍历所有标注
count = 0
for annotation in tqdm(data["annotations"]):
    count += 1
    image_id = annotation["image_id"]
    image_info = next((img for img in data["images"] if img["id"] == image_id), None)
    if not image_info:
        continue

    width, height = image_info["width"], image_info["height"]

    for idx, region in enumerate(annotation["regions"]):
        obj_id = region.get("order")  
        occ_rate = region.get("occlude_rate", 0)
        obj_name = region.get("name", "").strip().lower()

        # 跳过无意义类别
        if obj_name in meanless_classes:
            continue

        segmentation = region["segmentation"]
        points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]

        # 创建 amodal mask
        mask_amodal = Image.new("L", (width, height), 0)
        draw_amodal = ImageDraw.Draw(mask_amodal)
        draw_amodal.polygon(points, fill=255)
        mask_amodal_np = np.array(mask_amodal, dtype=np.uint8)

        # 保存 amodal mask
        amodal_path = os.path.join(output_amodal, f"{image_id}_{obj_id}_{count}.png")
        mask_amodal.save(amodal_path)
        amodalmask_log.write(f"{image_id}_{obj_id}_{count}.png\n")

        if occ_rate == 0:   # 不遮挡
            # 保存 vis mask（与 amodal 相同）
            vis_path = os.path.join(output_vismask, f"{image_id}_{obj_id}_{count}.png")
            mask_amodal.save(vis_path)
            vismask_log.write(f"{image_id}_{obj_id}_{count}.png" + "\n")

            # 提取前景图像区域（以 mask_amodal 作为 mask）
            image_path = os.path.join(image_root, image_info["file_name"])
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGBA")
                rgba = np.array(image)
                mask_bool = mask_amodal_np.astype(bool)
                rgba[~mask_bool] = 0
                foreground = Image.fromarray(rgba)
                fg_path = os.path.join(output_foreground, f"{image_id}_{obj_id}_{count}.png")
                foreground.save(fg_path)         #存进前景文件夹
                foreground_log.write(f"{image_id}_{obj_id}_{count}.png" + "\n")   #存前景TXT
            else:
                print(f"图像文件不存在: {image_path}")
            # 暂时不保存前景，按原逻辑
            # aLLLLZZZZZ = None
        else:
            # occlude_rate > 0，需要单独提取 visible_mask（RLE 格式）
            vis_rle = region.get("visible_mask", {})
            if vis_rle and "counts" in vis_rle:
                rle = {
                    "size": [height, width],
                    "counts": vis_rle["counts"].encode("utf-8")
                }
                vis_mask_np = maskUtils.decode(rle)
                vis_mask_img = Image.fromarray((vis_mask_np * 255).astype(np.uint8))
                vis_path = os.path.join(output_vismask, f"{image_id}_{obj_id}_{count}.png")
                vis_mask_img.save(vis_path)
                vismask_log.write(f"{image_id}_{obj_id}_{count}.png\n")
            else:
                print(f"可见区域 RLE 数据不存在: {image_id}_{obj_id}")

# 关闭日志文件
foreground_log.close()
vismask_log.close()
amodalmask_log.close()
print("数据集构建完成！（已过滤掉 meanless 类别）")
