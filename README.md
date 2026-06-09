<div align="center">

# 🛡️ Aegis
### AI-Powered Campus Safety & Surveillance System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00BFFF?style=flat-square)
![Claude API](https://img.shields.io/badge/Claude-LLM%20Reasoning-8A2BE2?style=flat-square)


*Behaviour-aware AI surveillance that translates raw visual data into actionable security intelligence.*

[Features](#-features) · [Architecture](#️-system-architecture) · [Getting Started](#-getting-started) · [Roadmap](#-roadmap)

</div>

---

##  The Problem

Traditional CCTV systems are fundamentally reactive and limited:

- **Operator overload** — humans cannot effectively monitor more than a few feeds at once
- **Delayed detection** — incidents are flagged after the fact, not as they unfold
- **Shallow analysis** — most systems detect *motion*, not *meaning*
- **No context** — alerts lack the reasoning needed for fast, confident responses

---

##  The Solution

Aegis is a **multi-model AI pipeline** that understands human behaviour in real time — not just what appears in frame, but what is *happening*.

It chains together computer vision models (object detection, pose estimation, action recognition) and feeds their combined output to a Claude LLM, which generates human-readable, severity-classified alerts that security teams can act on immediately.

**The result:** fewer missed incidents, faster response times, and explainable alerts — not just bounding boxes.

---

##  System Architecture

```
Video / Image Feed
        ↓
YOLOv8 Object Detection       →  Detects and localises people & objects
        ↓
Pose Estimation               →  Identifies posture anomalies (falls, collapse, crouching)
        ↓
Action Recognition Model      →  Classifies violent or suspicious interactions
        ↓
Claude LLM (Reasoning Layer)  →  Synthesises signals into a contextual, human-readable alert
        ↓
Severity Classification       →  EMERGENCY / HIGH / MEDIUM / LOW / CLEAR
        ↓
Actionable Security Alert     →  Logged, timestamped, exportable as PDF report
```

Each layer contributes a distinct type of understanding. No single model carries the full load — intelligence emerges from the pipeline as a whole.

---

##  Features

###  Image Analysis
- Object detection with labelled bounding boxes
- Pose-based fall and collapse detection
- Fight and violence classification
- AI-generated incident description with severity score

###  Video Analysis
- Frame-by-frame incident detection across the full clip
- Incident timeline with timestamped events
- Multi-frame evidence tracking to reduce false positives
- Severity-based event grouping for rapid triage

###  Multi-Camera Simulation
- Upload multiple feeds and assign each to a campus location
- Parallel analysis across all feeds simultaneously
- Grid-based results view for monitoring at a glance
- Per-camera incident logs and severity summaries

###  Reporting & Logging
- Persistent incident logging with structured JSON output
- PDF report generation for security documentation
- Explainable AI alerts — every alert includes the reasoning behind it
- Severity history for trend analysis

---

##  Example Scenarios

| Scenario | Detection Trigger | Severity | Response |
|----------|------------------|----------|----------|
| Students walking between classes | Normal movement, no anomaly | 🟢 CLEAR | No alert |
| Individual stationary for extended period | Pose + loitering detection | 🟡 MEDIUM | Monitoring alert |
| Two individuals in physical contact with aggressive motion | Action recognition | 🔴 HIGH | Immediate response alert |
| Person on ground, motionless | Pose estimation — collapse detected | 🚨 EMERGENCY | Medical response alert |

---

##  Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Dashboard | Streamlit | Web UI and camera feed management |
| Object Detection | YOLOv8 | Person and object localisation |
| Pose Estimation | OpenCV + HuggingFace Transformers | Posture and fall detection |
| Action Recognition | Custom trained model | Violence and behaviour classification |
| LLM Reasoning | Claude API (Anthropic) | Alert generation and contextual reasoning |
| Report Generation | ReportLab | PDF incident report export |
| Core Language | Python 3.10+ | Entire backend pipeline |

---

##  Project Structure

```
Aegis/
│
├── app.py                    # Main Streamlit dashboard & UI entry point
├── video_analysis.py         # Frame extraction & video processing pipeline
├── action_recognition.py     # Fight/violence detection model wrapper
├── requirements.txt          # Python dependencies
│
├── pipeline/                 # Core AI modules
│   ├── detector.py           # YOLOv8 object detection
│   ├── pose_estimator.py     # Pose & fall detection
│   └── llm_reasoning.py      # Claude API integration & alert generation
│
├── assets/                   # Static images & architecture diagrams
│
├── demos/                    # Sample test videos & images
│
├── notebooks/                # Research & model experimentation
│   └── model_eval.ipynb      # Accuracy benchmarking
│
└── incident_logs/            # Persistent storage for detection events
    └── *.json                # Per-incident structured log files
```

---

##  Getting Started

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/aegis.git
cd aegis

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

### Run

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

##  Current Limitations

| Limitation | Detail |
|-----------|--------|
| Pose integration | Video pose estimation not fully integrated across all pipelines |
| Live feeds | Uses uploaded footage — live RTSP streams not yet supported |
| Crowd density | Currently LLM-inferred, not model-driven |
| Deployment | Prototype-level — not production hardened |

---

## Roadmap

- [ ] Live CCTV (RTSP) stream integration
- [ ] Real-time edge deployment (Jetson Nano / similar)
- [ ] Dedicated crowd density detection model
- [ ] Improved temporal behaviour modelling across frames
- [ ] Multi-campus dashboard with role-based access
- [ ] Alert notification system (email / SMS / Slack)

---

##  Design Philosophy

> Aegis is not just a detection system — it is a **decision-support system** for security operations.

The goal is not to replace human judgment, but to augment it. By the time a security alert reaches an operator, Aegis has already done the heavy lifting: detecting, classifying, reasoning, and summarising — so the operator can focus entirely on *responding*.

---

