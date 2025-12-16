import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os


from dataset import OcclusionSegmentationDataset
import numpy as np
from unet import UNet
from swin_unet import SwinUNet
from PIL import Image
import yaml
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, recall_score

def calculate_accuracy(outputs, targets, fg_mask, threshold=0.5):    
    # outputs, targets, fg_mask 形状应为 [batch_size, channels, ...]
    # print("outputs shape:", outputs.shape)  # 检查输出形状
    # print("targets shape:", targets.shape)
    # print("fg_mask shape:", fg_mask.shape)
    batch_size = outputs.shape[0]
    total_acc = 0.0
    
    predicted = (outputs > threshold).float() 
    for i in range(batch_size):       
        # print("predicted shape:", predicted.shape)
        predicted_i = predicted[i,0]
        targets_i = targets[i,0]
        fg_mask_i = fg_mask[i,0]
        outputs_i = outputs[i,0]
        # print(predicted.shape)
        # print(fg_mask.shape)
        # correct = (predicted == targets).float()
        fg_mask_i = (fg_mask_i != 0)

        # fg_mask_bool = (fg_mask != 0) #*************用来可视化***************************

        fg_predicted_i = predicted_i[fg_mask_i]
        fg_targets_i = targets_i[fg_mask_i]
        # print("len(fg_predicted) = ",len(fg_predicted))
        # print("len(fg_targets) = ",len(fg_targets))
        # print("fg_predicted = ",fg_predicted)
        # print("fg_targets = ",fg_targets)
        
        
        # 计算前景区域内的准确率
        if fg_predicted_i.numel() == 0:  # 如果没有前景像素
            accuracy = 0
            # return 0.0
        else:        
            correct = (fg_predicted_i == fg_targets_i).float()
            accuracy = correct.mean()
        total_acc += accuracy

#*************用来可视化***************************
    # visualization = False

    # if visualization == True:
        
    #     # 创建可视化图像 (RGBA格式)
    #     height, width = outputs.shape[:2]
    #     visualization_result = np.ones((height, width, 4), dtype=np.uint8) * 255  # 初始化为白色
        
    #     # 定义颜色
    #     GREEN = np.array([0, 255, 0, 255])    # 预测和目标都是1 (真阳性)
    #     BLUE = np.array([0, 0, 255, 255])     # 预测和目标都是0 (真阴性)
    #     RED = np.array([255, 0, 0, 255])      # 预测和目标不一致 (假阳性/假阴性)
    #     # 获取前景区域的索引
    #     fg_indices = np.where(fg_mask_bool.cpu().numpy())
        
    #     # 为前景区域内的像素着色
    #     for i in range(len(fg_indices[0])):
    #         y, x = fg_indices[0][i], fg_indices[1][i]
    #         # print(predicted[y, x])
    #         # print(targets[y, x])
    #         if predicted[y, x] == 1 and targets[y, x] == 1:
    #             visualization_result[y, x] = GREEN  # 真阳性
    #         elif predicted[y, x] == 0 and targets[y, x] == 0:
    #             visualization_result[y, x] = BLUE   # 真阴性
    #         else:
    #             visualization_result[y, x] = RED    # 预测错误
        
    #     # 保存可视化结果
    #     Image.fromarray(visualization_result).save("temp/visualization_result.png")
    #     Image.fromarray((predicted*255).cpu().numpy().astype(np.uint8)).save("temp/result.png")
    #     print(f"可视化结果已保存")

#*************用来可视化***************************

    return total_acc

def calculate_mIoU(outputs, targets, fg_mask, threshold=0.5):
    # outputs, targets, fg_mask 形状应为 [batch_size, channels, ...]
    batch_size = outputs.shape[0]
    total_iou = 0.0
    # 处理输入数据，提取需要的部分
    # 遍历 batch 中的每个样本
    for i in range(batch_size):
        predicted = (outputs > threshold).float()
        predicted_i = predicted[i,0]
        targets_i = targets[i,0]
        fg_mask_i = fg_mask[i,0]
        
        # 确定前景区域
        fg_mask_i = (fg_mask_i != 0)
        
        # 提取前景区域的预测和目标值
        fg_predicted_i = predicted_i[fg_mask_i]
        fg_targets_i = targets_i[fg_mask_i]
        
        # 计算准确率
        if fg_predicted_i.numel() == 0:  # 如果没有前景像素
            iou = 0.0
        else:        
            # IoU计算
            # 计算交集 (TP: 预测为1且实际为1)
            intersection = (fg_predicted_i * fg_targets_i).sum()
            # 计算并集 (TP + FP + FN)
            union = fg_predicted_i.sum() + fg_targets_i.sum() - intersection
            # 计算IoU
            iou = intersection / union if union != 0 else 0.0
        total_iou += iou    
    return total_iou

def test_model(model, dataset, device, batch_size=8):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_miou = 0

    # 存储所有前景区域的预测和目标值
    all_fg_preds = []
    all_fg_targets = []
    
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)    #这里设置了batchsize
    with torch.no_grad():
        i = 0
        for inputs, fg_mask, targets in test_dataloader:
            inputs = inputs.to(device)
            fg_mask = fg_mask.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = F.binary_cross_entropy(outputs, targets, weight=fg_mask)
            
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, targets, fg_mask).item()  #张量中的单个数值提取出来，转换为 Python 原生类型
            total_miou += calculate_mIoU(outputs, targets, fg_mask).item()


            # 处理前景区域的预测和目标值
            # 将张量转换为numpy数组并展平
            outputs_np = outputs.detach().cpu().numpy().flatten()
            targets_np = targets.detach().cpu().numpy().flatten()
            fg_mask_np = fg_mask.detach().cpu().numpy().flatten()
            
            # 仅保留前景掩码为1的区域
            fg_indices = fg_mask_np == 1
            fg_preds = outputs_np[fg_indices]
            fg_targets = targets_np[fg_indices]
            
            # 收集所有批次的前景数据
            all_fg_preds.extend(fg_preds)
            all_fg_targets.extend(fg_targets)

            outputs_Image = Image.fromarray((outputs[0,0].cpu().numpy()*255).astype(np.uint8))
            i = i+1
            # outputs_Image.save("/pub/data/lz/autoComposition/autocomp_master/temp/"+str(i)+".png")

    # 计算平均损失和准确率    
    # print(f"----------------{len(test_dataloader)}----------------")  # 4
    avg_loss = total_loss / len(test_dataloader)
    avg_accuracy = (total_accuracy/batch_size) / len(test_dataloader)
    avg_miou = (total_miou/batch_size) / len(test_dataloader)
    # 计算前景区域的AP和AR
    # 处理可能没有前景像素的边缘情况
    if len(all_fg_targets) > 0:
        # 计算AP (使用原始预测概率)
        ap = average_precision_score(all_fg_targets, all_fg_preds)
        # 计算AR (使用0.5作为阈值进行二值化)
        ar = recall_score(all_fg_targets, np.array(all_fg_preds) > 0.5)
    else:
        ap = 0.0
        ar = 0.0
        print("警告: 未检测到前景像素，AP和AR将设为0")

    return avg_loss, avg_accuracy, avg_miou, ap, ar

def load_model(ckpt_path="/pub/data/lz/autoComposition/autocomp_master/ckpt/model_final.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
    if config['net']['module'] == "Unet":
        model = UNet()
    elif config['net']['module'] == "SwinUnet":
        model = SwinUNet() 
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.to(device)
    return model



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    test_dataset = OcclusionSegmentationDataset(root_dir=config['test']['dataset'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path=config['test']['ckpt_path'], device=device)
    avg_loss, avg_accuracy = test_model(model, test_dataset, nn.BCELoss(weight=None), device,config['test']['batch_size'])
    print("avg_loss = " ,avg_loss," ;   avg_accuracy = ", avg_accuracy)