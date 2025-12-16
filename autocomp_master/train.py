import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["WANDB_API_KEY"] = "e0e1e13374da76d1c29aa6e1d9b7be26dea9b7cb"
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"]="true"  #禁用wandb

import numpy as np
from dataset import OcclusionSegmentationDataset
from dataset_coco import OcclusionSegmentationDataset_coco
from test_dataset import Test_OcclusionSegmentationDataset
from dataset_already_have import AlreadyHaveBackDataset
from unet import UNet

from test import test_model

import yaml

import torch.nn.functional as F


def train_model(model, dataset, epochs=10, batch_size=4, lr=1e-4, project_name="my_project"):
    # 初始化 wandb
    wandb.init(project=project_name, settings=wandb.Settings(init_timeout=120),config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "architecture": model.__class__.__name__,
    })
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.BCELoss(weight=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
    train_loss_list = []
    test_loss_list = []
    test_accuracy_list = []
    test_ap_list = []
    test_ar_list = []

    best_test_acc = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0

        for step, (inputs, fg_mask, targets) in enumerate(train_dataloader):
            print("\r完成进度{0}".format(step / len(train_dataloader)), end="", flush=True)
            inputs = inputs.to(device)
            fg_mask = fg_mask.to(device)
            targets = targets.to(device)

            # foreground_mask = (inputs != 1).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss2 = criterion(outputs, targets) #这是全图BCELoss
            # criterion2 = nn.BCELoss(weight=fg_mask)
            # loss = criterion2(outputs, targets )  #前景内BCELoss

            # print(fg_mask.cpu().numpy()) #看似01变量
            # loss = F.binary_cross_entropy(outputs, targets, weight=fg_mask)
            loss_type = config['train'].get('loss_type', 'fg')  # 默认前景内loss

            if loss_type == "full":
                loss = F.binary_cross_entropy(outputs, targets)  # 全图loss
            elif loss_type == "fg":
                loss = F.binary_cross_entropy(outputs, targets, weight=fg_mask)  # 前景内loss
            else:
                raise ValueError(f"Unknown loss_type {loss_type}")

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()

            # 可视化部分样本（只取第一个 batch 第一个样本）
            if step == 0:
                input_img = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)
                target_img = targets[0].detach().cpu().numpy().squeeze()
                output_img = outputs[0].detach().cpu().numpy().squeeze()


                # Normalize if needed
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

                wandb.log({
                    "background": wandb.Image(input_img[:,:,:3], caption="Input"),
                    "foreground": wandb.Image(input_img[:,:,3:], caption="Input"),
                    "Target": wandb.Image(target_img, caption="Ground Truth"),
                    "Prediction": wandb.Image(output_img, caption="Prediction"),
                    "Prediction01": wandb.Image((output_img > 0.5), caption="Prediction01"),
                    "Loss": loss.item()
                }, step=epoch)

                

        train_avg_loss = train_total_loss / len(train_dataloader)
        train_loss_list.append(train_avg_loss)
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {train_avg_loss:.4f}")
        
        
        # 测试阶段
        test_avg_loss, test_avg_accuracy, test_avg_miou, test_ap, test_ar = test_model(model, test_dataset, device)
        test_loss_list.append(test_avg_loss)
        test_accuracy_list.append(test_avg_accuracy)
        test_ap_list.append(test_ap)
        test_ar_list.append(test_ar)

        if test_avg_accuracy > best_test_acc:
            best_test_acc  = test_avg_accuracy
            best_test_iou = test_avg_miou
            best_test_ap = test_ap
            best_test_ar = test_ar
            best_epoch = epoch

            
            if config['net']['use_depth'] == True:
                save_path = ckpt_root + "/best_" + config['net']['module']+"_"+ config['net']['merge_feat_method']+"_"+ config['train']['bg_mode']+"_"+config['train']['loss_type']+"_depth_final.pth"
            else:
                save_path = ckpt_root + "/best_" + config['net']['module']+"_"+ config['net']['merge_feat_method']+"_"+ config['train']['bg_mode']+"_"+config['train']['loss_type']+"final.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1} 模型权重已保存至 {save_path}")

        print(f"\nEpoch {epoch+1}/{epochs}, Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_avg_accuracy:.4f},Test mIoU:{test_avg_miou}, AP:{test_ap},AR:{test_ar}")

        # 每个 epoch 上传指标
        wandb.log({
            "Train Epoch Loss": train_avg_loss,
            "Test Epoch Loss": test_avg_loss,
            "Test Accuracy": test_avg_accuracy
        }, step=epoch)

        # # 保存每个 epoch 的模型
        # if epoch % 10 == 0 :
        #     save_path = ckpt_root + f"/model_epoch_{epoch+1}.pth"
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torch.save(model.state_dict(), save_path)
        #     print(f"Epoch {epoch+1} 模型权重已保存至 {save_path}")


        # 每个 epoch 上传 loss
        #wandb.log({"Epoch Loss": avg_loss}, step=epoch)

    # 创建保存目录
    save_dir = "/pub/data/lz/autoComposition/autocomp_master/temp"
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(test_accuracy_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    # 绘制AP曲线
    plt.subplot(2, 2, 3)
    plt.plot(test_ap_list, label='Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.title('AP Curve')
    plt.legend()
    plt.grid(True)
    
    # 绘制AR曲线
    plt.subplot(2, 2, 4)
    plt.plot(test_ar_list, label='Average Recall')
    plt.xlabel('Epoch')
    plt.ylabel('AR')
    plt.title('AR Curve')
    plt.legend()
    plt.grid(True)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()
    
    # 单独保存每个指标的曲线图（可选）
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'))
    plt.close()

    if config['net']['use_depth'] == True:
        save_path = ckpt_root + "/" + config['net']['module']+"_"+ config['net']['merge_feat_method']+"_depth_final.pth"
    else:
        save_path = ckpt_root + "/" + config['net']['module']+"_"+ config['net']['merge_feat_method']+"final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存至 {save_path}")
    print(f"encoder:{config['net']['module']},decoder:{config['net']['merge_feat_method']},best_test_acc = {best_test_acc}")
    best_metrics = {
        "best_acc": float(best_test_acc),
        "best_mIoU": float(best_test_iou),
        "best_AP": float(best_test_ap),
        "best_AR": float(best_test_ar),
        "best_epch":float(best_epoch),
    }
    with open("best_metrics.yaml", "w") as f:
        yaml.safe_dump(best_metrics, f)

    wandb.finish()
    return model

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    ckpt_root = config['train']['ckpt_root']   #好像贼不好的一种用法，定义了个函数里不可改的全局变量吧算是

    if config['net']['module'] == "Unet":
        model = UNet()
    elif config['net']['module'] == "SwinUnet":        
        from swin_depth_unet import SwinUNet
        # from swin_unet import SwinUNet
        model = SwinUNet() 

    print(f"NET: encoder={config['net']['module']},decoder = {config['net']['merge_feat_method']}")

    # train_dataset = OcclusionSegmentationDataset(root_dir=config['train']['dataset'])
    # test_dataset = Test_OcclusionSegmentationDataset(root_dir=config['test']['dataset'])

#拼接所有数据集进行训练--------------------------------------------------------------------------------------------------------------------------------
    # train_dataset1 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_basket_bottle")
    # train_dataset2 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_bottle_basket")
    # train_dataset3 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table")
    # train_dataset4 = OcclusionSegmentationDataset(root_dir=config['train']['dataset'])

    # from torch.utils.data import ConcatDataset
    # combined_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3, train_dataset4])

    
    
    # total_size = len(combined_dataset)  # 获取数据集总样本数
    # train_size = int(0.7 * total_size)  # 70%作为训练集
    # test_size = total_size - train_size  # 剩余30%作为测试集


    # # 随机分割数据集（设置seed确保可复现）
    # from torch.utils.data import random_split
    # torch.manual_seed(42)  # 固定随机种子，保证每次分割结果一致
    # train_dataset, test_dataset = random_split(
    #     dataset=combined_dataset,
    #     lengths=[train_size, test_size]
    # )

    train_dataset = OcclusionSegmentationDataset(root_dir=config['train']['dataset'])
    test_dataset = OcclusionSegmentationDataset_coco(root_dir="/pub/data/lz/autoComposition/autocomp_master/dataset/forComp_COCOA_occlude_filter_big")

    # test_dataset = Test_OcclusionSegmentationDataset(root_dir=config['test']['dataset'])
    trained_model = train_model(model, train_dataset, epochs=config['train']['epochs'], batch_size=config['train']['batch_size'], lr=config['train']['learning_rate'], project_name="first-train")



# #拆分篮子数据集进行训练--------------------------------------------------------------------------------------------------------------------------------
#     train_dataset1 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_basket_bottle")
#     train_dataset2 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_bottle_basket")
    
#     # train_dataset = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table")

#     from torch.utils.data import ConcatDataset
#     combined_dataset = ConcatDataset([train_dataset1, train_dataset2])
#     total_size = len(combined_dataset)  # 获取数据集总样本数
#     train_size = int(0.7 * total_size)  # 70%作为训练集
#     test_size = total_size - train_size  # 剩余30%作为测试集

#     # 随机分割数据集（设置seed确保可复现）
#     from torch.utils.data import random_split
#     torch.manual_seed(42)  # 固定随机种子，保证每次分割结果一致
#     train_dataset, test_dataset = random_split(
#         dataset=combined_dataset,
#         lengths=[train_size, test_size]
#     )
#     trained_model = train_model(model, train_dataset, epochs=config['train']['epochs'], batch_size=config['train']['batch_size'], lr=config['train']['learning_rate'], project_name="first-train")




# #拆分桌子凳子数据集进行训练--------------------------------------------------------------------------------------------------------------------------------
#     train_dataset = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table")
#     total_size = len(train_dataset)  # 获取数据集总样本数
#     train_size = int(0.7 * total_size)  # 70%作为训练集
#     test_size = total_size - train_size  # 剩余30%作为测试集

#     # 随机分割数据集（设置seed确保可复现）
#     from torch.utils.data import random_split
#     torch.manual_seed(42)  # 固定随机种子，保证每次分割结果一致
#     train_dataset, test_dataset = random_split(
#         dataset=train_dataset,
#         lengths=[train_size, test_size]
#     )
#     trained_model = train_model(model, train_dataset, epochs=config['train']['epochs'], batch_size=config['train']['batch_size'], lr=config['train']['learning_rate'], project_name="first-train")

# #拆分桌子数据集进行训练 瓶子测试--------------------------------------------------------------------------------------------------------------------------------
#     train_dataset = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table")
#     test_dataset = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_basket_bottle")      
#     trained_model = train_model(model, train_dataset, epochs=config['train']['epochs'], batch_size=config['train']['batch_size'], lr=config['train']['learning_rate'], project_name="first-train")


#拼接篮子凳子数据集进行训练 拆分测试--------------------------------------------------------------------------------------------------------------------------------
    # train_dataset1 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_basket_bottle")
    # train_dataset2 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_bottle_basket")
    # train_dataset3 = AlreadyHaveBackDataset(root_dir = "/pub/data/lz/autoComposition/autocomp_master/dataset/my_chair_table")
    # # train_dataset4 = OcclusionSegmentationDataset(root_dir=config['train']['dataset'])

    # from torch.utils.data import ConcatDataset
    # combined_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])

    
    # total_size = len(combined_dataset)  # 获取数据集总样本数
    # train_size = int(0.7 * total_size)  # 70%作为训练集
    # test_size = total_size - train_size  # 剩余30%作为测试集

    # # 随机分割数据集（设置seed确保可复现）
    # from torch.utils.data import random_split
    # torch.manual_seed(42)  # 固定随机种子，保证每次分割结果一致
    # train_dataset, test_dataset = random_split(
    #     dataset=combined_dataset,
    #     lengths=[train_size, test_size]
    # )
    # trained_model = train_model(model, train_dataset, epochs=config['train']['epochs'], batch_size=config['train']['batch_size'], lr=config['train']['learning_rate'], project_name="first-train")


