# Automated Traffic System (ATS) - Standalone Setup Guide

This project is an intelligent Automated Traffic Management System built with Flask, YOLOv8 for vehicle & emergency vehicle detection, and Hugging Face Vision Transformers (`tiya1012/vit-accident-image`) for accident detection and automated signal prioritization.

---

## 🚀 How to Run on Another Laptop

### 1. Prerequisites
- **Python 3.9+** (Python 3.10, 3.11, 3.12, or 3.13 recommended)
- **CUDA-compatible GPU** (optional, automatically uses CPU if GPU is not available)

---

### 2. Quick Setup

#### Step A: Open Terminal / Command Prompt in this folder
```bash
cd path/to/ATS_Standalone_Project
```

#### Step B: (Recommended) Create & Activate a Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
- **Mac / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### Step C: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

### 3. Start the Application
Run the Flask server:
```bash
python app.py
```

Once the models load, open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📁 Project Structure

```
ATS_Standalone_Project/
├── app.py                     # Main Flask Application & Traffic Controller
├── requirements.txt           # Python Dependencies
├── README.md                  # Instructions for setup
├── models/
│   ├── emergency_model.pt     # YOLO model for vehicles & emergency vehicles
│   └── accidentmodel.pt       # Legacy weights
├── templates/
│   └── index.html             # Dashboard UI
├── static/
│   ├── css/
│   │   └── style.css          # Styling & Animations
│   ├── js/
│   │   └── main.js            # Telemetry polling & chart controllers
│   └── outputs/               # Real-time annotated frame outputs
└── uploads/                   # Upload storage for input images/videos
```

---

## ✨ Features Included
- **Exact Incident Keyframe Extraction**: Scans multi-frame video uploads to pinpoint the exact frame where an accident or emergency vehicle occurs.
- **Vision Transformer (ViT) Accident Detection**: Uses `tiya1012/vit-accident-image` for high-accuracy crash classification.
- **YOLO Emergency Vehicle Priority**: Detects ambulances, fire trucks, and police vehicles, instantly clearing green corridors.
- **Dynamic Signal Timing**: Allocates 30s base time + 2s per detected vehicle (up to 90s).
- **Manual & Pedestrian Overrides**: Force green signals or trigger safe 5-second pedestrian crossing phases.
"# Automated-Traffic-System" 
