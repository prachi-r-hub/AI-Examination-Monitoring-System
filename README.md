# AI-Assisted Examination Monitoring and Restricted Object & Behaviour Detection Using Computer Vision

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2Bcu118-orange.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Hardware](https://img.shields.io/badge/NVIDIA-GeForce%20GTX%201050-76B900.svg)](https://www.nvidia.com/)

---

## 📌 1. Project Title
**AI-Assisted Examination Monitoring and Restricted Object & Behaviour Detection Using Computer Vision**

---

## ❓ 2. Problem Statement
Manual examination proctoring in academic institutions faces significant operational challenges:
- **Human Fatigue**: Invigilators monitoring large examination halls or multiple video feeds experience attention degradation over extended durations.
- **Subtle Contraband Usage**: Small electronic devices (such as smartwatches, earbuds, or compact smartphones) can easily be concealed during examinations.
- **Accusation Risks & False Positives**: Manual observation without quantitative audit trails can lead to subjective disputes between candidates and invigilators.

There is a critical need for an **automated, objective, AI-assisted examination monitoring system** that aids human invigilators by detecting candidate objects, tracking spatial persistence, flagging observable suspicious behaviour, and maintaining an immutable audit log—**without ever making direct cheating accusations**.

---

## 🎯 3. Objective
The primary objective of this project is to build a robust, real-time computer vision system that:
1. **Detects & Categorizes Examination Objects**: Automatically identifies 7 target custom classes, categorizing them into **ALLOWED** stationary stationery (`Notebook`, `Paper`, `Pen`, `Pencil`) versus **RESTRICTED** contraband (`Phone`, `Smartwatch`, `Earbuds`).
2. **Filters Transient Noise via Spatial Persistence**: Enforces consecutive-frame confirmation ($N \ge 3$) to eliminate false alarms from single-frame detection noise.
3. **Tracks Object Duration & Association**: Measures continuous handling duration (`duration_seconds`) and spatially links restricted items to nearby candidate `person_id` tracks.
4. **Detects Observable Suspicious Behaviours**: Identifies predefined behavioral patterns including repeated lateral head movements and spatial-temporal object/paper passing between candidates.
5. **Computes Transparent Monitoring Risk**: Evaluates a rule-based 0–100 monitoring risk score with explicit point breakdowns for invigilator review.
6. **Provides Non-Blocking Audio & Evidence Capture**: Dispatches daemon thread voice warnings and captures annotated high-confidence evidence screenshots (`incident_XXX.jpg`) linked to an SQLite database.
7. **Delivers a Professional Control-Room Web Application**: Presents a modern React + Vite frontend served by a high-performance FastAPI backend.

---

## 🏗️ 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       React + Vite Frontend (UI Layer)                      │
│   Nav: 1. Monitoring  2. Image  3. Video  4. Incidents  5. Evidence  6. Settings│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST API + MJPEG Video Stream
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Python Backend                            │
│  ├── GET /api/camera/stream  (MJPEG Single Camera Video Stream)             │
│  ├── POST /api/camera/start  (Single VideoCapture(0) initialization)       │
│  ├── POST /api/camera/stop   (Release VideoCapture(0) cleanly)              │
│  ├── GET  /api/camera/status (Real-time metrics: FPS, risk, detections)     │
│  ├── POST /api/image/detect  (Single Image YOLOv8s Detection)               │
│  ├── POST /api/video/analyze (Video Upload & Full Processor Pipeline)      │
│  ├── GET/PATCH /api/incidents (SQLite DB Records & Status Updates)          │
│  ├── GET  /api/evidence   (Captured Screenshot Files & Metadata)            │
│  ├── GET/POST /api/settings (System Thresholds & Evaluation Suite)          │
│  └── GET  /api/export/csv   (Incident Database CSV Export)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Reuses Existing Modules
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Existing Backend Modules                          │
│  ├─ detector.py (ExamDetector, best.pt)  ├─ risk_engine.py (RiskEngine)     │
│  ├─ tracker.py (IncidentTracker)         ├─ voice_alert.py (VoiceAlert)     │
│  ├─ person_tracker.py (PersonTracker)   ├─ evidence_manager.py (Evidence)  │
│  ├─ head_detector.py & passing_detector  ├─ database.py (SQLite DB)         │
│  └─ video_processor.py (VideoProcessor) └─ evaluator.py (ResearchEval)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ 5. YOLOv8s Custom Object Detection Engine
- **Model Architecture**: YOLOv8s (Small) pretrained backbone fine-tuned on custom examination dataset (`best.pt`).
- **Input Resolution**: Optimized at $640 \times 480$ for NVIDIA GTX 1050 4GB VRAM safety.
- **Inference Performance**: ~33.6 ms latency per frame on CUDA GPU (~29.8 FPS raw inference).
- **Class Classification Layer**: Encapsulated within `detector.py:ExamDetector` to filter invalid labels automatically.

---

## 📊 6. Custom Dataset
- **Dataset Source**: `Classroom_Dataset/Merged_Dataset` containing 1,031 validation images across 7 classes.
- **Split Structure**: `train/`, `valid/`, `test/` splits with YOLO format bounding box annotations.
- **Dataset Grounding**: Validation metrics evaluated directly on physical dataset annotations (`data.yaml`).

---

## 🏷️ 7. Seven Custom Classes
The model detects 7 custom classes:
1. `Notebook` (Class ID 0)
2. `Paper` (Class ID 1)
3. `Pen` (Class ID 2)
4. `Pencil` (Class ID 3)
5. `Phone` (Class ID 4)
6. `Smartwatch` (Class ID 5)
7. `Earbuds` (Class ID 6)

> [!IMPORTANT]
> **Dataset Anomaly Note**: An illegal class label `6` present in legacy phone dataset subfolders was identified and excluded from the final application architecture.

---

## 🟢 8. Allowed vs Restricted Objects

The system strictly divides detected objects into two safety categories:

| Category | Classes Included | Visual Bounding Box Style | System Action |
| :--- | :--- | :--- | :--- |
| **🟢 ALLOWED** | `Notebook`, `Paper`, `Pen`, `Pencil` | Solid Green (`#16a34a`) with label e.g. `pen 91% \| ALLOWED` | Processed as normal stationary examination tools. Generates **0 alerts** and **0 DB records**. |
| **🔴 RESTRICTED** | `Phone`, `Smartwatch`, `Earbuds` | Solid Red (`#dc2626`) with label e.g. `phone 96% \| RESTRICTED` | Subject to spatial persistence tracking ($N \ge 3$). Confirmed presence triggers alerts & DB logging. |

---

## 📷 9. Mode 1 — Image Detection
- **Functionality**: Upload custom examination images or select test dataset samples.
- **Display**: Side-by-side original image vs annotated YOLOv8s bounding box display.
- **Breakdown Table**: Displays object name, category, confidence score, bounding box coordinates, and center point.

---

## 🎥 10. Mode 2 — Live Webcam Monitoring (Single Camera Lifecycle)
- **Functionality**: Serves a single, dominant live camera stream directly from `backend/camera_manager.py`.
- **Controls**: `START CAMERA` (initializes single `cv2.VideoCapture(0)` background thread) and `STOP CAMERA` (cleanly releases hardware device).
- **Gauges**: 6 real-time metrics (`⚡ FPS`, `👤 People Detected`, `🟢 Allowed Objects`, `🔴 Restricted Objects`, `🚨 Confirmed Incidents`, `📈 Monitoring Risk`).
- **Active Table & Timeline**: Live active frame detections table and recent incident timeline.

---

## 🎬 11. Mode 3 — Pre-Recorded Video Analysis
- **Functionality**: Primary presentation mode for video file examination analysis (MP4, AVI, MOV, MKV).
- **Outputs**:
  - **Video Metrics**: Filename, duration, frames processed, processing time, actual FPS, GPU inference latency.
  - **Event Summary**: Counts for restricted objects, restricted handling, head movements, and object passing.
  - **Risk Summary**: Breakdown of Low, Medium, High risk events and overall session risk score ($0–100\%$).
  - **Chronological Timeline**: Clickable timeline of all detected observable events.
  - **Evidence Gallery**: Grid of captured annotated screenshot cards.
  - **Interactive Inspector**: Detailed event breakdown and proctor review controls.

---

## 🔄 12. Consecutive-Frame Confirmation
- **Purpose**: Prevents false alarms caused by single-frame model misclassifications or momentary noise.
- **Tracking Algorithm**: Spatial IoU ($	ext{IoU} \ge 0.30$) and Euclidean center distance ($	ext{Dist} \le 80\text{px}$) matching across frames.
- **Confirmation Rule**: A candidate restricted object MUST persist across **$N \ge 3$ consecutive frames** before triggering an incident record or voice alert.

---

## ⏱️ 13. Duration Tracking
- **Parameters**: Tracks `first_seen_time`, `last_seen_time`, and continuous `duration_seconds`.
- **Significance Categorization**:
  - `LOW` ($< 2.0\text{s}$): Brief presence / transient view.
  - `MEDIUM` ($2.0\text{s} - 5.0\text{s}$): Restricted-object handling detected.
  - `HIGH` ($> 5.0\text{s}$): Extended restricted-object handling detected.

---

## 👤 14. Person Tracking & Spatial Association
- **Module**: `person_tracker.py:PersonTracker`.
- **Pretrained Model**: Pretrained COCO `yolov8s.pt` filtered specifically for Class 0 (`person`).
- **Spatial Association**: Assigns restricted items to nearby candidate tracks based on bounding box proximity ($\le 80\text{px}$ buffer), labelling items as `Phone -> Person #1`.
- **Modular Toggle**: Can be toggled on/off on demand to optimize performance.

---

## 🚨 15. Restricted-Object Handling Detector
- **Event Wording**: `"Restricted-object handling detected (Phone, P#1, 3.2s)"`.
- **Trigger Criteria**: Confirmed restricted object ($N \ge 3$) associated with candidate track meeting duration threshold ($T \ge 2.0\text{s}$).

---

## 👤 16. Repeated Head-Movement Detector
- **Module**: `head_detector.py:HeadMovementDetector`.
- **Mechanism**: Estimates coarse head orientation (`FORWARD`, `LEFT`, `RIGHT`) per candidate track.
- **Trigger Criteria**: Detects repeated lateral orientation sequences (e.g. `FORWARD` $\rightarrow$ `LEFT` $\rightarrow$ `RIGHT`) exceeding threshold count ($N \ge 3$) within a $4.0\text{s}$ time window.
- **Event Wording**: `"Possible repeated head movement — candidate turning direction repeatedly"`.

---

## 📄 17. Possible Object Passing Detector
- **Module**: `passing_detector.py:ObjectPassingDetector`.
- **Mechanism**: Tracks spatial-temporal transfer patterns where an item moves from proximity of Person A ($P_A$) to Person B ($P_B$) and remains near $P_B$.
- **Conservatism**: Enforces low-confidence suppression ($< 0.30$) and duplicate event cooldowns.
- **Event Wording**: `"Possible object passing — review recommended"`.

---

## 📈 18. Transparent Monitoring Risk Engine
- **Module**: `risk_engine.py:RiskEngine`.
- **Scoring Scale**: Rule-based score from $0$ to $100$:
  - `LOW` ($0 - 29$): Normal monitoring state.
  - `MEDIUM` ($30 - 59$): Attention required / candidate handling.
  - `HIGH` ($60 - 100$): Confirmed warning / multiple risk factors.
- **Starting Factor Contributions**:
  - Restricted Object Base: **+30 pts**
  - High Detection Confidence ($\ge 0.85$): **+15 pts**
  - Long Continuous Duration ($\ge 5.0\text{s}$): **+20 pts**
  - Repeated Head Movement: **+20 pts**
  - Object/Paper Passing: **+25 pts**

---

## 🔊 19. Non-Blocking Voice Alerts
- **Module**: `voice_alert.py:VoiceAlertManager`.
- **Thread Safety**: Uses an asynchronous daemon thread to run `pyttsx3` speech synthesis without freezing YOLO inference, video processing, or UI loops ($< 1.0\text{ms}$ dispatch overhead).
- **Alert Texts**:
  - Restricted Object: `"Warning. Restricted object detected."`
  - Suspicious Behaviour: `"Suspicious activity detected. Please review."`
- **Cooldown**: Configurable $10.0\text{s}$ suppression cooldown prevents rapid duplicate audio alerts.

---

## 📸 20. Automatic Evidence Screenshot Capture
- **Module**: `evidence_manager.py:EvidenceManager`.
- **Frame Selection**: Uses a rolling 7-frame buffer to select the highest confidence frame for capture.
- **Annotation Overlay**: Draws bounding boxes, candidate labels, and a top dark header bar (`EVENT: ... | RISK: XX/100 [LEVEL] | TIME: ...`).
- **Storage**: Saves files to `evidence/live_incidents/` or `evidence/video_incidents/` as `incident_XXX.jpg` and links the file path to the database record.

---

## 💾 21. SQLite Incident Database
- **Module**: `database.py:IncidentDatabase`.
- **Storage File**: `database/incidents.db`.
- **Schema Fields**: `incident_id`, `timestamp`, `event_type`, `object_name`, `person_id`, `confidence`, `duration`, `risk_score`, `risk_level`, `evidence_path`, `review_status`, `notes`, `created_at`.
- **Review Actions**: Updates `review_status` (`Unreviewed`, `Reviewed`, `Dismissed`) and saves persistent invigilator notes.
- **Deduplication**: Prevents duplicate row insertions for continuous spatial tracks within a 5-second window.

---

## 🖥️ 22. Examination Control-Room Web Application
- **Frontend Architecture**: React + Vite single-page application served directly by FastAPI.
- **Design Aesthetic**: Professional Control Room styling (subtle 1px slate borders `#334155`, neutral dark slate background `#0f172a`, state-communicating colors GREEN/AMBER/RED).
- **Navigation**: 6 Primary Screens:
  1. `🎥 Monitoring` (Default main live webcam monitoring screen)
  2. `📷 Image Detection` (Single image upload & verification)
  3. `🎬 Video Analysis` (Uploaded video processing & report)
  4. `📋 Incidents` (SQLite database review & CSV export)
  5. `🖼️ Evidence` (Captured screenshot gallery & modal viewer)
  6. `⚙️ Settings` (System thresholds & research evaluation suite)

---

## ⚙️ 23. Installation & Setup Instructions

### Prerequisites
- Operating System: Windows 10/11
- Python: Python 3.12 64-bit
- Node.js: Node.js 18+ & npm
- GPU: NVIDIA GPU with CUDA support (e.g. GeForce GTX 1050 4GB)

### Step-by-Step Setup

1. **Activate Virtual Environment**:
   ```powershell
   cd "c:\Users\Vaibhav\OneDrive\Desktop\Object Recognition (minor project)"
   .\venv312\Scripts\Activate.ps1
   ```

2. **Install FastAPI & Dependencies**:
   ```powershell
   pip install fastapi uvicorn python-multipart
   ```

3. **Build Frontend React Production Bundle**:
   ```powershell
   cd frontend
   npm install
   npm run build
   cd ..
   ```

---

## 🚀 24. How to Run

### Option 1: Run Full Production Control Room (FastAPI + React Static Production Build)
```powershell
.env312\Scripts\python.exe backend/server.py
```
*Open your web browser at **`http://localhost:8000`**.*

### Option 2: Run Development Mode (FastAPI Backend + Vite Dev Server)
1. **Start FastAPI Backend (Terminal 1)**:
   ```powershell
   .env312\Scripts\python.exe backend/server.py
   ```
2. **Start Vite Dev Server (Terminal 2)**:
   ```powershell
   cd frontend
   npm run dev
   ```
*Open your web browser at **`http://localhost:5173`**.*

---

## 🧪 25. Unit & Integration Testing
```powershell
# Run 59 Unit Tests across 13 Modules
.env312\Scripts\python.exe -m unittest discover -p "test_*.py"
```

---

## 📊 26. Measured Evaluation & Research Reporting

Validation metrics evaluated on `Merged_Dataset/valid` (**1,031 images**) using **NVIDIA GeForce GTX 1050 (4GB)**:

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Precision ($P$)** | **72.10%** | Ratio of true positive detections over total positive predictions |
| **Recall ($R$)** | **63.92%** | Ratio of true positive detections over total ground truth instances |
| **mAP@50** | **67.40%** | Mean Average Precision at IoU threshold 0.50 |
| **mAP@50-95** | **52.67%** | Mean Average Precision averaged across IoU range 0.50–0.95 |
| **GPU Inference Latency** | **33.6 ms** | Average CUDA forward pass time per frame |
| **Live Throughput FPS** | **22.0 FPS** | Complete live pipeline execution frame rate on GTX 1050 |

---

## ⚠️ 27. System Limitations
1. **Lighting & Occlusion**: Low ambient classroom lighting or severe physical object occlusion can reduce detection confidence below the threshold ($0.25$).
2. **Coarse Head Orientation**: Head turning heuristics estimate 2D orientation bounding changes rather than full 3D facial mesh geometry.
3. **Camera Resolution**: Optical detection of tiny items (such as single earbuds) requires at least 720p webcam resolution at close proximity.

---

## 🔮 28. Future Work
1. **3D Head Pose Mesh Estimation**: Integration of MediaPipe 468-point 3D facial landmark mesh for precise pitch/yaw/roll head pose tracking.
2. **Multi-Camera Stream Synchronization**: Orchestration of multiple IP network cameras covering wide examination halls simultaneously.
3. **Edge AI Hardware Deployment**: Optimizing model quantization (TensorRT FP16/INT8) for standalone edge deployment on NVIDIA Jetson Orin Nano modules.
