# ablation_runner.py
import os
import subprocess
import yaml

# 定义实验组合
experiments = [
    #Swin+cat
    {"name": "swin_cat", "net": {"module": "SwinUnet", "merge_feat_method": "cat", "use_depth": True},"loss": "fg","bg_mode": "multi"},
    #Swin+crossatt
    {"name": "swin_cross", "net": {"module": "SwinUnet", "merge_feat_method": "crossattention", "use_depth": False},"loss": "fg","bg_mode": "single"},  
    #+multi bg  
    {"name": "swin_cross_multi_bg", "net": {"module": "SwinUnet", "merge_feat_method": "crossattention", "use_depth": False},"loss": "full", "bg_mode": "multi"},
    #+loss
    {"name": "swin_cross_loss_fg", "net": {"module": "SwinUnet", "merge_feat_method": "crossattention", "use_depth": False}, "loss": "fg", "bg_mode": "multi"},
    #+depth
    {"name": "swin_cross_depth", "net": {"module": "SwinUnet", "merge_feat_method": "crossattention", "use_depth": True}, "loss": "fg", "bg_mode": "multi"},
    # {"name": "swin_cross_depth_loss_fg", "net": {"module": "SwinUnet", "merge_feat_method": "crossattention", "use_depth": True}, "loss": "fg"},
    {"name": "unet", "net": {"module": "Unet", "merge_feat_method": "cat", "use_depth": True}, "loss": "fg", "bg_mode": "multi"},
]

results_file = "ablation_results.txt"

for exp in experiments:
    # 1. 更新 config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["net"].update(exp["net"])

    # if "loss" in exp:
    config["train"]["loss_type"] = exp["loss"]  # full / fg
    # if "bg_mode" in exp:
    config["train"]["bg_mode"] = exp["bg_mode"]  # multi / single

    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # 2. 调用 train.py
    print(f"\n===== Running experiment: {exp['name']} =====\n")
    subprocess.run(["python", "train.py"])

    # 3. 假设 train.py 结束时会打印最佳结果，写到 results.txt
    with open(results_file, "a") as f:
        f.write(f"Experiment: {exp['name']}\n")
        # 这里假设 train.py 运行过程中会把 best_test_acc, mIoU, AP, AR 保存到临时文件 best_metrics.yaml
        if os.path.exists("best_metrics.yaml"):
            with open("best_metrics.yaml", "r") as mf:
                metrics = yaml.safe_load(mf)
                f.write(str(metrics) + "\n\n")
