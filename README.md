# TrueFluence — Comprehensive Multimodal Scam & Deepfake Detection Platform

TrueFluence is a full-stack AI platform designed to detect fraudulent influencer campaigns, deepfakes, and scam videos. It evaluates content authenticity by fusing visual quality, audio consistency, deepfake heuristics, and NLP-driven engagement analysis. 

The platform consists of a mobile app, web dashboard, backend API, and a robust Multimodal AI engine.

---

## 🏗️ Project Architecture

* **`Multimodals/`** — The Core AI Engine (PyTorch, Transformers, MesoNet)
* **`MobileApp/`** — React Native (Expo) mobile application featuring an Instagram-like video feed
* **`Frontened/`** — Web-based user interface dashboard
* **`Backend/`** — Flask API connecting the mobile/web interfaces to the AI engine

---

## 🧠 The AI Pipeline (Multimodals)

The TrueFluence Multimodal engine processes videos through a strict **5-Step Sequential Pipeline** to calculate a final "Trust Score" (0.0 to 1.0).

### 1. MesoNet Deepfake Gate 🔍
* **Model**: Meso-4 (MesoNet architecture for Deepfake Detection)
* **Function**: Extracts frames and calculates a deepfake probability.
* **Gate Rule**: If the deepfake threshold exceeds **80%**, the pipeline **aborts immediately** and outputs a final score of `0.0` (⛔ DEEPFAKE).

### 2. Video Analysis Engine 🎥
* **MobileNetV2 Backbone**: Frozen ImageNet-pretrained feature extraction.
* **Quality Assessment Head**: Multi-layer MLP scoring video production quality per frame.
* **Temporal LSTM & Attention**: Bidirectional 2-layer LSTM and self-attention mechanism evaluating frame sequences over time.

### 3. Audio Analysis Engine 🎵
* **Feature Extraction**: 128-dim vectors combining MFCCs, Chroma, Spectral, RMS, and Mel Bands.
* **VGGish & Pattern Analysis**: Evaluates voice authenticity, pause anomalies, and audio-visual consistency.

### 4. Video + Audio Fusion 🔗
* **Weight**: Contributes **40%** to the final overarching score.
* **Architecture**: Concatenation-based MLP merging 135-dim visual and audio vectors. (Defaults to a penalty if the video lacks an audio track).

### 5. Comments & Engagement Engine 💬
* **Weight**: Contributes **60%** to the final overarching score (as social proof is a massive indicator of scam campaigns).
* **NLP (BERT)**: Uses `bert-base-uncased` from Hugging Face Transformers to assess comment sentiment, detecting bot rings and warnings from real users.
* **Engagement MLP**: Custom neural network weighing followers, likes, and comment volume ratios.
* **Dynamic Integration**: Automatically pulls live engagement data by reading a matching `<video_name>.json` file during testing.

---

## 📊 Verdict Confidence Zones

| Score Range | Verdict | Emoji | Action Required |
|---|---|---|---|
| 0.0 – 0.3 | **SCAM / DEEPFAKE** | 🔴 | High alert. Immediate takedown. |
| 0.3 – 0.5 | **LIKELY SCAM** | 🟠 | Highly suspicious. |
| 0.5 – 0.7 | **LIKELY REAL** | 🟡 | Borderline content. |
| 0.7 – 1.0 | **REAL** | 🟢 | Safe and authentic. |

---

## 🛠️ Installation & Setup

### AI Engine (Multimodals) Setup

```bash
cd Multimodals
# 1. Provide a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install core ML requirements
pip install -r requirements.txt

# 3. Ensure Transformers is installed (Required for Comments Engine)
pip install transformers

# 4. Initialize directories and download required MesoNet weights
python setup_project.py
```

---

## 🎯 Usage

### 1. Testing Videos
Place your target test videos in `Multimodals/Test_Dataset/`. Optionally, provide a `.json` file with the exact same name (e.g., `test_vid1.json`) containing engagement data:
```json
{
  "followers": 50000,
  "likes": 5200,
  "comments": ["Amazing quality!", "Is this a scam?", "Not working."]
}
```

Run the pipeline:
```bash
cd Multimodals
python test.py
```
*Results will print beautifully in your terminal and save to `results.txt` and `results.json`.*

### 2. Training the Core Engine
The visual and audio systems are trained sequentially across 4 phases to prevent catastrophic forgetting. Wait until one phase finishes before the next begins.
```bash
python train.py
```
*(Ensure `dataset/real_videos/` and `dataset/scam_videos/` are populated prior to training).*

---

## ⚠️ Known Limitations
* **Small Dataset Dependence**: The visual/audio components were base-trained on a smaller dataset; generalization may vary on unseen environments.
* **Transformers Requirement**: The Comments Engine will silently fall back to a `0.5` neutral score if `transformers` is not installed properly in your environment.