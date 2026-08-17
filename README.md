#  Spatial-Temporal Object Tracking Pipeline (YOLOv5 & LSTM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)
![YOLO](https://img.shields.io/badge/YOLOv5-Object%20Detection-00FFFF)

##  Project Overview
This repository contains an advanced hybrid Object Tracking pipeline. It combines spatial feature extraction using **YOLOv5** with temporal video sequence processing via a custom **LSTM network**. 

The primary objective is to resolve multi-object detection noise by effectively isolating and continuously tracking a single specific target across video frames, transitioning from mere object detection to continuous object tracking.

##  Technical Architecture

* **Spatial Feature Extraction:** Utilizes Ultralytics YOLOv5 to detect bounding boxes and extract spatial features from individual frames.
* **Temporal Processing:** A custom LSTM network (256 hidden dimension) takes 10-frame sequences as input to predict and smooth the object's trajectory.
* **Custom IoU Filtering Algorithm:** Implements a custom Intersection over Union (IoU) logic to filter out false positives from YOLO's multi-class predictions, ensuring the system stays locked on the correct target.
* **Data Engineering (`OTBDataLoader`):** A custom data parsing class built to ingest the OTB100 benchmark dataset, dynamically extracting and formatting video frames, applying Gaussian blur noise reduction via OpenCV, and synchronizing ground truth bounding boxes.

##  Repository Structure

```text
yolo-lstm-object-tracking/
│
├── data/                                # (Not tracked) Directory for OTB100 sequences
├── weights/                             # (Not tracked) YOLOv5 and LSTM weights
├── 01_main_yolo_lstm_tracking.ipynb     # Main tracking pipeline and architecture setup
├── 02_tracking_basketball_seq.ipynb     # Inference & evaluation on the BasketBall sequence
├── 03_tracking_blurcar_seq.ipynb        # Inference & evaluation on the BlurCar2 sequence
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore configurations
└── README.md                            # Project documentation
```

##  Installation & Prerequisites

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/yolo-lstm-object-tracking.git
   cd yolo-lstm-object-tracking
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup (Crucial):**
   This project was originally developed on Kaggle using the OTB100 benchmark. 
   * Download the relevant sequences (e.g., `BasketBall`, `BlurCar2`) from the official OTB100 dataset.
   * Place the extracted image sequences and their corresponding ground truth `.txt` files into the `data/` directory.

##  Usage

To run the pipeline or evaluate specific sequences, open the corresponding Jupyter Notebooks.

For example, to test the tracking on the Basketball sequence:
```bash
jupyter notebook 02_tracking_basketball_seq.ipynb
```
*(Ensure that paths inside the notebook point to your local `data/` directory instead of `/kaggle/input/...`)*

##  Author
**Mohammed Essalhi**
* [LinkedIn](https://linkedin.com/in/mohammed-essalhi-23794b24b)
