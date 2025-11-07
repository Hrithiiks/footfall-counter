# 🧠 Footfall Counter using Computer Vision

This project implements a **footfall counting system** using **YOLOv8** for person detection and **ByteTrack** for object tracking.  
It counts the number of people **entering** and **exiting** through a defined **Line of Interest (LOI)** using video footage from the **VIRAT Dataset**.

---

## 📘 Project Overview

- Detects and tracks people across video frames.  
- Counts **Entries** and **Exits** based on movement across a defined line.  
- Handles occlusion using **ByteTrack** multi-object tracking.  
- Outputs an annotated video and a CSV summary of counts.

---

## ⚙️ Technologies Used

| Component | Technology |
|------------|-------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Object Tracking | ByteTrack |
| Programming Language | Python 3.10 |
| Libraries | OpenCV, NumPy, PyYAML, tqdm |
| Dataset | VIRAT Dataset |

---

## 🧩 Project Structure
```bash
footfall-counter/
│
├── configs/
│ ├── virat.yaml # Video path, line coordinates, confidence threshold
│ └── bytetrack.yaml # ByteTrack tracker configuration
│
├── data/
│ └── virat_dataset/ # VIRAT dataset videos
│
├── outputs/
│ ├── videos/ # Annotated output videos
│ ├── counters/ # CSV summary files
│ └── screenshots/ # Screenshots for README visuals
│
├── src/
│ ├── app.py # Main script (YOLO + ByteTrack + counting)
│ └── drawing_loi.py # Optional script for drawing Line of Interest
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Hrithiiks/footfall-counter.git
cd footfall-counter
```
```bash
conda create -n footfall-counter python=3.10 -y
conda activate footfall-counter
```
```bash
pip install -r requirements.txt
```

```bash
python src/app.py
```

### 5️⃣ View Outputs

Annotated video → outputs/videos/

CSV summary → outputs/counters/


## 🖼️ Results

| People Detection | Entry/Exit Counting |
|------------------|--------------------|
| <img width="600" alt="frame1" src="https://github.com/user-attachments/assets/09a09d34-bca3-4b90-a55e-5b3cbb0bd159" /> | <img width="600" alt="frame2" src="https://github.com/user-attachments/assets/f1327d36-8e35-4737-8614-d10f2f837a6f" /> |


## 🧠 Future Improvements

Support multiple entrances using Region of Interest (ROI) masks

Integrate webcam or live CCTV feed

Add a web dashboard for real-time analytics and visualization

## 📚 Dataset Credits

This project uses video samples from the VIRAT Video Dataset
Developed by SRI International and collaborators for activity recognition and surveillance research.

📎 Dataset Link: https://viratdata.org/

© VIRAT Project — used strictly for research and educational purposes only.
