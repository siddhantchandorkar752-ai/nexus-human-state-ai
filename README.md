@'
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=NEXUS%20AI&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Human%20State%20Intelligence%20System&descAlignY=58&descSize=20" width="100%"/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00D9FF&center=true&vCenter=true&width=750&lines=Real-time+Emotion+Detection+%F0%9F%A7%A0;Posture+%2B+Spine+Analysis+%F0%9F%A7%8D;Heart+Rate+from+Camera+%E2%9D%A4%EF%B8%8F;Action+Unit+FACS+System+%F0%9F%91%81%EF%B8%8F;Live+on+HuggingFace+%F0%9F%A4%97" alt="Typing SVG"/>

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/siddhantchandorkar/nexus-human-state-ai)
[![GitHub](https://img.shields.io/badge/GitHub-nexus--human--state--ai-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai/nexus-human-state-ai)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 🧠 What is NEXUS?

NEXUS is a **research-level Human State Intelligence AI** that analyzes a person's complete mental and physical state in real-time using only a camera — no wearables, no sensors.
```python
class NEXUS:
    capabilities = [
        "👁️  Eye Blink Rate + Drowsiness Detection",
        "😤  FACS Action Unit Emotion System (7 emotions)",
        "🧍  Posture + Spine Alignment Scoring",
        "❤️  rPPG Heart Rate from Skin Color",
        "🎯  Optical Flow Micro-movement Stress",
        "🚨  Real-time Risk Scoring (LOW/MEDIUM/HIGH)",
        "🤖  YOLOv11 Person + Pose Detection"
    ]
```

---

## 🚀 Live Demo

<div align="center">

### 👉 [Try NEXUS on HuggingFace](https://huggingface.co/spaces/siddhantchandorkar/nexus-human-state-ai)

| Upload your photo or use webcam → Get instant analysis |
|---|

</div>

---

## 🔬 How It Works
```
Camera Feed
    │
    ├── YOLOv11 ──────────── Person Detection + Pose Keypoints
    │                              │
    │                         Posture Score
    │                         Spine Alignment
    │                         Blink Detection
    │
    ├── Action Units (FACS) ─ AU1,AU4,AU6,AU12,AU15,AU17,AU20,AU23,AU43
    │                              │
    │                         7 Emotions: Happy/Sad/Angry/Fear/Surprise/Disgust/Neutral
    │
    ├── rPPG ─────────────── Green channel skin signal → Heart Rate BPM
    │
    └── Optical Flow ──────── Micro-movements → Stress Score
                                    │
                              ┌─────▼─────┐
                              │ RISK ENGINE│
                              │ LOW/MED/HIGH│
                              └───────────┘
```

---

## 📊 System Output

| Signal | Method | Output |
|---|---|---|
| 👁️ Eye State | EAR from pose keypoints | OPEN / CLOSED |
| 😤 Emotion | FACS Action Units | 7 emotions + confidence |
| 🧍 Posture | Keypoint geometry | Score /100 + issues |
| ❤️ Heart Rate | rPPG green channel | BPM |
| 🎯 Stress | Optical flow + fusion | 0.0 — 1.0 |
| 🚨 Risk | Weighted ensemble | LOW / MEDIUM / HIGH |

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## ⚡ Quick Start
```bash
git clone https://github.com/siddhantchandorkar752-ai/nexus-human-state-ai
cd nexus-human-state-ai
pip install -r requirements.txt
python nexus.py
```

For video file:
```bash
python nexus.py --source video.mp4
```

---

## 📁 Project Structure
```
nexus-human-state-ai/
├── nexus.py                  ← Main inference (webcam/video)
├── app.py                    ← Gradio web app
├── models/
│   ├── vision_core.py        ← YOLOv11 + rPPG + Blink + Posture
│   ├── emotion_engine.py     ← FACS Action Units → 7 Emotions
│   └── risk_scorer.py        ← Weighted risk fusion
├── Dockerfile
└── requirements.txt
```

---

<div align="center">

**Built with 🔥 by [Siddhant Chandorkar](https://github.com/siddhantchandorkar752-ai)**

[![](https://visitcount.itsvg.in/api?id=nexus-human-state-ai&icon=6&color=6)](https://visitcount.itsvg.in)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&text=NEXUS%20%E2%80%94%20See%20Beyond%20the%20Surface&fontSize=18&fontColor=fff&animation=twinkling" width="100%"/>

</div>
'@ | Set-Content -Path README.md -Encoding UTF8
