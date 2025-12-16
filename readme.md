# Place Anywhere: Learning Spatial Reasoning for Occlusion-aware Image Composition

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C.svg)](https://pytorch.org/)

## 📝 Introduction
This repository contains the implementation for **Place Anywhere: Learning Spatial Reasoning for Occlusion-aware Image Composition**. The project focuses on automatic object composition using datasets such as BSDS-A and leverages methods like `pix2gestalt` for foreground generation.
<img width="1337" height="431" alt="framework_all" src="https://github.com/user-attachments/assets/a6d5d26b-a394-4494-9661-679a2520d03d" />


## 🛠️ Environment Setup

### Reference Hardware
The code has been developed and tested on the following high-performance computing environment:
* **OS:** Ubuntu 22.04
* **CPU:** AMD EPYC 7763 @ 2.45GHz (256 cores)
* **RAM:** 512GB
* **GPU:** 3x NVIDIA Tesla A100
* **IP:** 192.168.20.10 (Internal)

### Software Requirements
* **Python:** 3.8.8
* **PyTorch:** 1.12.1+cu113
* **Torchaudio:** 0.12.1+cu113
* **Torchvision:** 0.13.1+cu113

To install the necessary dependencies, please run:

```bash
conda create -n autocomp python=3.8.8
conda activate autocomp
pip install -r requirements.txt
```
## 📂 Data Preparation
- 1. Download Source Data
First, download the BSDS-A dataset.

- 2. Generate Masks and Foregrounds
This project uses a two-step process to prepare the data.

Step 1: Extract basic masks and foregrounds

```Bash
python getForeBackMask_from_BSDSA.py
```
This script extracts partial complete foregrounds and two types of masks.

Step 2: Generate full foregrounds using Pix2Gestalt Note: Ensure you activate the pix2gestalt environment for this step.
```Bash
conda activate pix2gestalt
python getFullFore_by_pix2gestalt.py
```

Uses pix2gestalt method to obtain all foregrounds.
3. Directory Structure
After preparation, your directory structure should look like this:
```Plaintext
autocomp_master
└── dataset/
    ├── BSDS/
    ├── BSDSAandCOCOA/
    └── forComp_dataset/
        ├── foreground/
        ├── objmask_amodal/
        ├── objmask_visiable/
        ├── amodalmask.txt
        ├── foreground_from_generate.txt
        ├── foreground.txt
        └── vismask.txt
```
4. Dataset Splits
Please ensure you select the correct dataset configuration in the code/args:


## 🚀 Training
To train the model, navigate to the project root and run the training script:

```Bash
cd autoComposition/autocomp_master
python train.py
```

## 🧪 Testing
To evaluate the model on the test dataset:

```Bash
python test.py
```
## 💻 Demo / Inference
We provide a web-based interface for easy inference.
Start the inference server:

```Bash
python inference.py
```
Open the URL displayed in your terminal (e.g., http://127.0.0.1:7860) in your web browser.

<img width="1103" height="1080" alt="demo示意" src="https://github.com/user-attachments/assets/73dbff9b-ae33-478d-94c1-f6df0fea5eac" />


