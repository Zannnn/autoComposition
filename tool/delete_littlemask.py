import os
from PIL import Image
import numpy as np

def calculate_mask_ratio(image_path):
    """计算掩码图中掩码区域占全图的比例"""
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            total_pixels = img_array.size
            
            if len(img_array.shape) == 3:
                mask_pixels = np.count_nonzero(img_array.any(axis=2))
            else:
                mask_pixels = np.count_nonzero(img_array)
            
            ratio = mask_pixels / total_pixels if total_pixels > 0 else 0
            return min(ratio, 1.0)
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")
        return None

def find_low_ratio_files(mask_folder, threshold=0.05):
    """找出掩码占比低于阈值的文件列表"""
    low_ratio_files = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    
    for file in os.listdir(mask_folder):
        file_path = os.path.join(mask_folder, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
            ratio = calculate_mask_ratio(file_path)
            if ratio is not None and ratio < threshold:
                low_ratio_files.append(file)
    
    return low_ratio_files

def delete_corresponding_files(source_files, target_folder):
    """删除目标文件夹中与源文件列表同名的文件"""
    deleted_count = 0
    for file in source_files:
        # target_folder = "/pub/data/lz/autoComposition/otherMethod/results/GT_amodal"
        target_path = os.path.join(target_folder, file)
        if os.path.exists(target_path) and os.path.isfile(target_path):
            try:
                os.remove(target_path)
                deleted_count += 1
                print(f"已删除: {target_path}")
            except Exception as e:
                print(f"删除文件 {target_path} 失败: {str(e)}")
        else:
            print(f"文件不存在: {target_path}")
    
    return deleted_count

def main():
    # 文件夹路径设置
    root_path = "/pub/data/lz/autoComposition/autocomp_master/dataset/forComp_COCOA_occlude_filter_big/"
    mask_folder = root_path + "objmask_amodal"
    foreground_folder = root_path + "foreground"
    threshold = 0.02  # 5%的阈值
    
    # 检查文件夹是否存在
    if not os.path.exists(mask_folder):
        print(f"错误: 掩码文件夹 {mask_folder} 不存在")
        return
    
    if not os.path.exists(foreground_folder):
        print(f"错误: 前景文件夹 {foreground_folder} 不存在")
        return
    
    # 查找低占比掩码文件
    print(f"开始查找掩码占比小于{threshold*100}%的文件...")
    low_ratio_files = find_low_ratio_files(mask_folder, threshold)
    
    # 显示统计信息
    print(f"\n找到 {len(low_ratio_files)} 个掩码占比小于{threshold*100}%的文件:")
    for file in low_ratio_files[:10]:  # 只显示前10个文件名
        print(f"  {file}")
    if len(low_ratio_files) > 10:
        print(f"  ... 还有 {len(low_ratio_files)-10} 个文件")
    
    # 确认删除操作
    if low_ratio_files:
        confirm = input(f"\n是否要删除前景文件夹中对应的 {len(low_ratio_files)} 个文件? (y/n): ")
        if confirm.lower() == 'y':
            deleted = delete_corresponding_files(low_ratio_files, foreground_folder)
            print(f"\n操作完成，共删除 {deleted} 个文件")
        else:
            print("已取消删除操作")
    else:
        print("\n没有找到符合条件的文件，无需删除")

if __name__ == "__main__":
    main()
    