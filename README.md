# Multi-Modal Pedestrian Crossing Prediction <br>with the PIE Dataset

Predict whether a pedestrian will **cross the street in the near future** by fusing multiple information streams from the **Pedestrian Intention Estimation (PIE)** dataset.

<div align="center">
  <img src="https://github.com/user-attachments/assets/9c10d389-6996-471d-9279-e483a23a80c4" alt="Model overview" width="650">
</div>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add a license badge -->

This repository hosts the code for a multi-stream LSTM-based framework designed to predict pedestrian crossing behavior. It utilizes the Pedestrian Intention Estimation (PIE) dataset and explores the integration of diverse data modalities, including bounding box trajectories, skeletal pose, panoptic scene features, pedestrian behaviors, ego-vehicle dynamics, and static environmental context.

## 🌟 Key Features

*   **Multi-Modal Integration:** Leverages synchronized streams from bounding boxes, pose, panoptic segmentation (YOLOP), pedestrian actions, ego-vehicle data, and environment.
*   **Robust Data Pipeline:** Efficiently processes the PIE dataset, including annotation consolidation and feature standardization.
*   **Advanced Feature Extraction:**
    *   Pedestrian Pose: Using **YOLOv8x-pose-p6**.
    *   Panoptic Scene Features: Using **YOLOP** for drivable area, lane lines, and nearby objects.
    *   PIE Native: Bounding boxes, pedestrian behaviors (action, look, occlusion), ego-vehicle data, traffic light states, and static attributes.
*   **Data Balancing:** Employs oversampling with Gaussian noise augmentation for the minority "crossing" class.
*   **Flexible Model Architecture:** A Multi-Stream LSTM model where each stream is processed by a dedicated Bidirectional LSTM and an Attention mechanism.
*   **Fusion Techniques:** Explores both simple concatenation and a learnable Weighted Average Fusion of stream context vectors.
*   **Systematic Evaluation:** Supports ablation studies to assess the contribution of different streams and fusion techniques using F1-score, Precision, Recall, and AUC.

## 📂 Repository Structure
```
├── JAAD/                 # Contains the notebook for using the best model for inference on the JAAD dataset 
│
├── Offline Extraction/       # Contains the notebooks for extracting the pose data and the panoptic scene features 
│   ├── yolo_pose_extraction.ipynb  
│   └── yolop-extraction.ipynb       
│
├── Previous Versions/       # Contains the previous version of the model creation and ablation study
│
├── Final Version         # Contains the version used to create the best model           
│
└── README.md
```


## ⚙️ Setup and Installation
(Recommended to use kaggle to run the code instead of locally)

### 1. Prerequisites
*   Git
*   Python 3.8+
*   Anaconda or Miniconda (recommended for managing environments)

### 2. Clone Repositories
```bash
# Clone this project
git clone https://github.com/samalouty/MSLSTM-PID.git
cd MSLSTM-PID

# Clone PIE dataset utilities (essential for data parsing)
git clone https://github.com/aras62/PIE.git PIE

# Clone YOLOP (if running YOLOP feature extraction yourself)
git clone https://github.com/hustvl/YOLOP.git YOLOP
```

### 3. Download Datasets and Pretrained Weights
PIE Dataset:
Videos: Download the PIE video clips (set01 to set06) and place them in a directory (e.g., data/pie_videos/set01/video_0001.mp4).
Update VIDEO_INPUT_DIR in Final Version.ipynb to point to this directory.

Annotations: Download annotations.zip, annotations_attributes.zip, annotations_vehicle.zip from the PIE GitHub or their official download links. Unzip them into the PIE/annotations/ directory, resulting in structures like PIE/annotations/annotations/set01/....

Model Weights:

YOLOv8x-pose-p6: The pose extraction notebook (Offline Extraction/yolo_pose_extraction.ipynb) will attempt to download this automatically via ultralytics.

YOLOP Weights: If running YOLOP feature extraction, download End-to-end.pth (or yolop.pth) from the YOLOP releases and place it in YOLOP/weights/. Update the path in the Offline Extraction/yolop-extraction.ipynb if needed.

(Optional) Pre-extracted Features:

If you have pre-extracted features, place them in the directories specified by POSE_DATA_DIR (for pose) and YOLOP_FEATURE_DIR (for YOLOP features).

If you have a pre-generated pie_database.pkl, place it where PIE_DATABASE_CACHE_PATH points.

Generate them using the PIE utility files if you do not have them. 
