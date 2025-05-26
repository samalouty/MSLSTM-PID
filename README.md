# Multi-Stream LSTM for Pedestrian Intention Prediction

Predict whether a pedestrian will **cross the street in the near future** by fusing multiple information streams from the **Pedestrian Intention Estimation (PIE)** dataset.

<div align="center">
  <img src="https://github.com/user-attachments/assets/9c10d389-6996-471d-9279-e483a23a80c4" alt="Model overview" width="650">
</div>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository hosts the code for a multi-stream LSTM‑based framework designed to predict pedestrian crossing behavior. It utilizes the Pedestrian Intention Estimation (PIE) dataset and integrates diverse data modalities, including bounding‑box trajectories, skeletal pose, panoptic scene features, pedestrian behaviors, ego‑vehicle dynamics, and static environmental context.

## 🌟 Key Features

* **Multi‑Modal Integration** – Synchronized streams from bounding boxes, pose, panoptic segmentation (**YOLOP**), pedestrian actions, ego‑vehicle data, and static environment.  
* **Robust Data Pipeline** – Consolidates PIE annotations and standardises features.  
* **Advanced Feature Extraction**  
  * **YOLOv8x‑pose‑p6** for pedestrian pose.  
  * **YOLOP** for drivable area, lane lines, and nearby‑object context.  
  * PIE‑native streams: bounding boxes, pedestrian behaviours (action, look, occlusion), ego‑vehicle OBD, traffic‑light state, static attributes.  
* **Data Balancing** – Oversamples the minority *crossing* class with Gaussian‑noise augmentation.  
* **Flexible Model Architecture** – Each stream processed by a Bidirectional LSTM followed by an Attention layer.  
* **Fusion Techniques** – Simple concatenation **or** learnable weighted‑average fusion of stream context vectors.  
* **Systematic Evaluation** – Supports ablations; metrics reported include F1, precision, recall, and AUC.

## 📂 Repository Structure
```text
├── JAAD/                  # Inference notebook for the JAAD dataset
│
├── Offline Extraction/    # Notebooks for pose & panoptic feature extraction
│   ├── yolo_pose_extraction.ipynb
│   └── yolop-extraction.ipynb
│
├── Previous Versions/     # Early experiments & ablation studies
│
├── Final Version/         # Notebook that trains the best model
│
└── README.md
```

## ⚙️ Setup & Installation
> **Tip:** Running on **Kaggle** (GPU P100) is often easier than a local setup.

### 1&nbsp;· Prerequisites
* Git
* Python ≥ 3.8
* **conda/miniconda** (recommended)

### 2&nbsp;· Clone Repositories
```bash
# Main project
git clone https://github.com/samalouty/MSLSTM-PID.git
cd MSLSTM-PID

# PIE dataset utilities (XML parsing helpers)
git clone https://github.com/aras62/PIE.git PIE

# YOLOP (only if you need to re‑extract scene features)
git clone https://github.com/hustvl/YOLOP.git YOLOP
```

### 3&nbsp;· Download Datasets & Pre‑Trained Weights

#### PIE videos  
Download *set01* … *set06* clips and place them, e.g.
```
data/pie_videos/set01/video_0001.mp4
```
Update `VIDEO_INPUT_DIR` in **Final Version.ipynb**.

#### PIE annotations  
Unzip `annotations.zip`, `annotations_attributes.zip`, `annotations_vehicle.zip` into:
```
PIE/annotations/annotations/set01/...
```

#### Model weights
| Component | Where to get it | Destination |
|-----------|-----------------|-------------|
| **YOLOv8x‑pose‑p6** | `ultralytics` hub – downloaded automatically by *yolo_pose_extraction.ipynb* | — |
| **YOLOP** (`End-to-end.pth`) | YOLOP repo releases | `YOLOP/weights/` |

#### (Optional) Pre‑extracted features
If you already have `.pkl` files for pose or YOLOP, place them where `POSE_DATA_DIR` and `YOLOP_FEATURE_DIR` point.  
A cached `pie_database.pkl` can also be dropped at `PIE_DATABASE_CACHE_PATH`.

### 4&nbsp;· Python Environment
```bash
# Conda example
conda create -n mslstm_pid_env python=3.9 -y
conda activate mslstm_pid_env

# PyTorch (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Remaining deps
pip install numpy pandas scikit-learn tqdm matplotlib seaborn opencv-python ultralytics pyyaml jupyterlab yacs
```

---

## 🚀 Running the Code
The workflow is notebook‑driven:

1. **Offline Feature Extraction** *(optional but recommended)*  
   *Run once to create pose and YOLOP `.pkl` files.*
   * `Offline Extraction/yolo_pose_extraction.ipynb`  
   * `Offline Extraction/yolop-extraction.ipynb`

2. **Training & Evaluation – `Final Version.ipynb`**  
   *Phase A – Data preparation* (auto‑runs if caches missing):  
   parses XML, builds `pie_database.pkl`, computes scalers, and balances classes.  
   *Phase B – Model*  
   set `ACTIVE_STREAMS` to choose modalities:
   ```python
   ACTIVE_STREAMS = [
       "bbox",
       "ped_action",
       "ped_look",
       # "pose",          # ← enable to add pose
       "ego_speed",
       "ego_acc",
       "yolop",    # YOLOP drivable/lane
   ]
   ```
   Train for `NUM_EPOCHS`; best model (by F1) is saved automatically.

3. **Ablation Studies**  
   Repeat step 2 with different `ACTIVE_STREAMS` lists and compare results.

---

## 📊 Expected Outputs
| Artifact | Purpose |
|----------|---------|
| `scalers.pkl` | Mean / std params for numeric features |
| `aug_balanced_train_data_with_features.pkl` | Augmented training split |
| `best_model_[streams]_ep[X].pth` | Model checkpoint |
| Notebook stdout | Epoch‑wise metrics |
| Notebook figures | Loss curves, confusion matrix |

---

## 🙏 Acknowledgements & Citation
If you use the PIE dataset, please cite:
```bibtex
@inproceedings{rasouli2019pie,
  title      = {{PIE}: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction},
  author     = {Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K},
  booktitle  = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages      = {6262--6271},
  year       = {2019}
}
```

If you use YOLOP, please cite:
```bibtex
@article{wu2020yolop,
  title   = {{YOLOP}: You Only Look Once for Panoptic Driving Perception},
  author  = {Wu, Dong and Liao, Manwen and Zhang, Weitian and Wang, Xinggang and Bai, Xiang and Cheng, Wenqing and Liu, Wenyu},
  journal = {arXiv preprint arXiv:2108.11250},
  year    = {2021}
}
```

---
