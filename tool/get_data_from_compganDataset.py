import os
from PIL import Image

# 路径配置

# things_to_deal = "image"
things_to_deal = "image"

if things_to_deal == "image":
    src_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/chair_table/paired/images"
    back_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table/images/back"
    fore_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table/images/fore"
elif things_to_deal == "annotation":
    src_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/chair_table/unpaired/images"
    back_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table/annotation/back"
    fore_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table/annotation/fore"


# 确保目标文件夹存在
os.makedirs(back_dir, exist_ok=True)
os.makedirs(fore_dir, exist_ok=True)

# 遍历源文件夹
for filename in os.listdir(src_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        src_path = os.path.join(src_dir, filename)
        try:
            img = Image.open(src_path)
            w, h = img.size
            slice_width = w // 5  # 每份宽度

            # 最右边 1/5
            back_crop = img.crop((w - slice_width, 0, w, h))
            back_crop.save(os.path.join(back_dir, filename))

            # 倒数第二右边 1/5
            fore_crop = img.crop((w - 2 * slice_width, 0, w - slice_width, h))
            fore_crop.save(os.path.join(fore_dir, filename))

            print(f"处理完成: {filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

print("所有图片处理完成。")
