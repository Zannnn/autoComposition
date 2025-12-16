import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_iou(pred_mask, gt_mask):
    """计算两个掩码的IoU"""
    # 确保两个掩码形状相同
    assert pred_mask.shape == gt_mask.shape, "掩码形状必须相同"
    
    # 计算交集和并集
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # 避免除以零
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_miou(baseline_folder, gt_folder):
    """计算所有图像对的平均IoU (mIoU)"""
    # 获取所有图像文件名
    image_files = [f for f in os.listdir(baseline_folder) 
                  if os.path.isfile(os.path.join(baseline_folder, f))]
    
    total_iou = 0.0
    count = 0
    
    for file in tqdm(image_files, desc="计算mIoU"):
        # 构建文件路径
        baseline_path = os.path.join(baseline_folder, file)
        gt_path = os.path.join(gt_folder, file)
        
        # 检查GT文件是否存在
        if not os.path.exists(gt_path):
            print(f"警告: GT文件 {file} 不存在，已跳过")
            continue
        
        # 读取图像并转换为二值掩码 (0或1)
        try:
            baseline_mask = np.array(Image.open(baseline_path).convert('L')) > 127
            gt_mask = np.array(Image.open(gt_path).convert('L')) > 127
            
            # 计算IoU并累加
            iou = calculate_iou(baseline_mask, gt_mask)
            total_iou += iou
            count += 1
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}，已跳过")
    
    if count == 0:
        return 0.0
    
    return total_iou / count

def calculate_accuracy(baseline_folder, gt_folder, gt_amodal_folder):
    """
    计算特定accuracy: 
    GT_amodal所在像素中，baseline和GT相同的像素占GT_amodal所在像素的比例
    """
    # 获取所有图像文件名
    image_files = [f for f in os.listdir(baseline_folder) 
                  if os.path.isfile(os.path.join(baseline_folder, f))]
    
    total_matching = 0
    total_amodal_pixels = 0
    
    for file in tqdm(image_files, desc="计算Accuracy"):
        # 构建文件路径
        baseline_path = os.path.join(baseline_folder, file)
        gt_path = os.path.join(gt_folder, file)
        amodal_path = os.path.join(gt_amodal_folder, file)
        
        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [baseline_path, gt_path, amodal_path]):
            print(f"警告: 至少一个文件 {file} 不存在，已跳过")
            continue
        
        try:
            # 读取图像并转换为二值掩码 (0或1)
            baseline_mask = np.array(Image.open(baseline_path).convert('L')) > 127
            gt_mask = np.array(Image.open(gt_path).convert('L')) > 127
            amodal_mask = np.array(Image.open(amodal_path).convert('L')) > 127
            
            # 确保所有掩码形状相同
            assert baseline_mask.shape == gt_mask.shape == amodal_mask.shape, \
                f"文件 {file} 的掩码形状不一致"
            
            # 找到GT_amodal所在的像素位置
            amodal_pixels = amodal_mask == 1
            
            # 计算这些位置中baseline和GT相同的像素数量
            matching_pixels = np.logical_and(
                amodal_pixels, 
                np.logical_xor(baseline_mask, gt_mask) == 0  # 相同为True
            ).sum()
            
            # 累加结果
            total_matching += matching_pixels
            total_amodal_pixels += amodal_pixels.sum()
            
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}，已跳过")
    
    if total_amodal_pixels == 0:
        return 0.0
    
    return total_matching / total_amodal_pixels

def main():
    # 三个文件夹的路径，可以根据实际情况修改
    # root_path = "/pub/data/lz/autoComposition/otherMethod/results_bsdsa/"
    # baseline_folder = root_path + "baseline"
    # gt_folder = root_path + "GT"
    # gt_amodal_folder = root_path + "GT_amodal"

    
    root_path = "/pub/data/lz/autoComposition/otherMethod/results_bsdsa/"
    baseline_folder = root_path + "Tan"
    gt_folder = root_path + "GT"
    gt_amodal_folder = root_path + "GT_amodal"
    
    # root_path = "/pub/data/lz/autoComposition/autocomp_master/dataset/forComp_BSDSA_occlude_filter_big/"
    # baseline_folder = root_path + "objmask_amodal"
    # gt_folder = root_path + "objmask_visiable"
    # gt_amodal_folder = root_path + "objmask_amodal"
    
    # 检查文件夹是否存在
    for folder in [baseline_folder, gt_folder, gt_amodal_folder]:
        if not os.path.exists(folder):
            print(f"错误: 文件夹 {folder} 不存在!")
            return
    
    # 计算mIoU
    miou = calculate_miou(baseline_folder, gt_folder)
    print(f"\nbaseline与GT的mIoU: {miou:.4f}")
    
    # 计算accuracy
    accuracy = calculate_accuracy(baseline_folder, gt_folder, gt_amodal_folder)
    print(f"GT_amodal区域内baseline与GT的匹配率: {accuracy:.4f}")

if __name__ == "__main__":
    main()
