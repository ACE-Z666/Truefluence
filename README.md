# 🛡️ TrueFluence
> **A Comprehensive Multimodal Scam & Deepfake Detection Platform**

TrueFluence is a full-stack, production-ready AI architecture designed to autonomously detect fraudulent influencer campaigns, deepfakes, and scam videos. By aggregating asynchronous inference gates, feature extraction across multiple modalities, and an advanced multimodal fusion layer, TrueFluence ensures content authenticity at scale. 

---

## 📁 Directory Architecture

TrueFluence adopts a decoupled, modular design to ensure scalable engineering habits, clean code separation, and rapid local or production deployment.

```text
📦 TrueFluence
 ┣ 📂 apps/                   # Client Applications
 ┃ ┣ 📂 mobile-app/           # React Native (Expo) mobile app featuring an Instagram-like video feed
 ┃ ┗ 📂 web-dashboard/        # Modern single-page HTML/CSS social credibility scanner and report viewer
 ┣ 📂 core-ai/                # Core AI Engine (PyTorch, MesoNet, Transformers)
 ┃ ┣ 📂 comments/             # BERT Comments Classifier & Engagement MLP training/inference modules
 ┃ ┣ 📂 config/               # Pipeline configuration and parameters
 ┃ ┣ 📂 src/                  # Pipeline engine core source modules (video, audio, MesoNet, NLP, fusion)
 ┃ ┣ 📂 Test_Dataset/         # Test reels, inference reports (results.txt, results.json)
 ┃ ┗ 📜 test_inference.py     # Main CLI test script to run end-to-end evaluation
 ┃ ┗ 📜 train_pipeline.py     # Multi-phase sequential pipeline training harness
 ┣ 📂 server/                 # Flask REST API backend connecting client interfaces to the AI engine
 ┃ ┣ 📂 api/                  # API endpoints and route designs
 ┃ ┣ 📂 uploads/              # Transient storage for uploaded video scans
 ┃ ┗ 📜 app.py                # Main backend application runner serving both REST API and web-dashboard
 ┗ 📂 Test_videos/            # Empty placeholder directory for additional user-provided test inputs
```

---

## 🏗️ Multimodal AI Pipeline (System Design)

The core AI engine evaluates content via a strict **5-Step Sequential Pipeline** to calculate a final predictive "Trust Score" (0.0 to 1.0). 


### 1. MesoNet Deepfake Gate 🔍 (`core-ai/src/mesonet.py`)
* **Architecture**: Meso-4 (MesoNet architecture for Deepfake Detection).
* **Functionality**: Serves as a high-speed asynchronous inference gate extracting frames to calculate deepfake probability.
* **Gate Rule**: If the deepfake fraction of frames exceeds **80%**, the pipeline aborts immediately and outputs a final score of `0.0` (⛔ DEEPFAKE), preventing unnecessary compute downstream.

### 2. Video Analysis Engine 🎥 (`core-ai/src/video_engine.py`)
* **Backbone**: MobileNetV2 for frozen ImageNet-pretrained feature extraction.
* **Quality Assessment**: Multi-layer MLP scoring video production quality per frame.
* **Temporal Analysis**: Bidirectional 2-layer LSTM and self-attention mechanism evaluating frame sequences over time for semantic consistency.

### 3. Audio Analysis Engine 🎵 (`core-ai/src/audio_engine.py`)
* **Feature Extraction**: 128-dim vectors combining MFCCs, Chroma, Spectral, RMS, and Mel Bands.
* **Pattern Analysis**: Utilizes VGGish embeddings to evaluate voice authenticity, pause anomalies, and audio-visual consistency.

### 4. Multimodal Fusion Layer 🔗 (`core-ai/src/fusion_layer.py`)
* **Weighting**: Contributes **40%** of the final overarching score.
* **Architecture**: Concatenation-based MLP merging 135-dim visual and audio vectors. Defaults to an architectural penalty if the video lacks an audio track.

### 5. Comments & Engagement Engine 💬 (`core-ai/src/nlp_engine.py` & `core-ai/comments/`)
* **Weighting**: Contributes **60%** of the final overarching score (capitalizing on social proof as a primary indicator of scam campaigns).
* **NLP (BERT)**: Deploys `bert-base-uncased` from Hugging Face Transformers to assess comment sentiment, detecting bot rings and real-user warnings.
* **Engagement Analytics**: Custom neural network weighing followers, likes, and comment volume ratios. Automatically integrates live engagement data via a matching `<video_name>.json` file.

---

## 📊 Verdict Confidence Matrix

| Score Range | Verdict | Indicator | Required Action |
| :--- | :--- | :---: | :--- |
| **0.0 – 0.3** | **SCAM / DEEPFAKE** | 🔴 | High alert. Immediate takedown required. |
| **0.3 – 0.5** | **LIKELY SCAM** | 🟠 | Highly suspicious. Flag for manual review. |
| **0.5 – 0.7** | **UNCERTAIN** | 🟡 | Borderline content. Monitor engagement. |
| **0.7 – 1.0** | **REAL** | 🟢 | Safe and authentic. |

---

## 🛠️ Quickstart & Local Deployment

### 1. AI Engine (`core-ai`) Setup

```bash
cd core-ai

# Provision an isolated virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install core machine learning dependencies
pip install -r requirements.txt

# Initialize project directory structure and populate dummy dataset
python setup_project.py
```

#### Data Ingestion & Inference
Place target test videos in `core-ai/Test_Dataset/`. For social context inference, optionally provide a matching `.json` file (e.g., `test_vid1.json`):

```json
{
  "followers": 50000,
  "likes": 5200,
  "comments": [
    "Amazing quality!", 
    "Is this a scam?", 
    "Not working."
  ]
}
```

Execute the command line evaluation pipeline:
```bash
python test_inference.py
```
> *Outputs are formatted in the terminal and serialized to `Test_Dataset/results.txt` and `Test_Dataset/results.json`.*

#### Pipeline Training
The visual and audio systems undergo sequential training across 4 phases to mitigate catastrophic forgetting.
```bash
python train_pipeline.py
```
> *Ensure `dataset/real_videos/` and `dataset/scam_videos/` are hydrated with training data prior to execution.*

---

### 2. Flask REST Backend (`server`) Setup

The backend serves as the bridge between client apps and the underlying AI models. It also hosts the web dashboard statically.

```bash
cd server

# Install backend dependencies (Flask, Flask-Cors)
pip install -r requirements.txt

# Start the server on port 5000
python app.py
```
> *The server will start listening at `http://localhost:5000`.*

---

### 3. Client Dashboards & Apps (`apps`) Setup

#### A. Web-Based Dashboard (`apps/web-dashboard`)
* Fully responsive and interactive dashboard built in HTML, CSS, and vanilla JS.
* Displays analyzed reels, provides real-time progress bars for each model bucket, and highlights deepfake blocks.
* **Access**: Simply open `http://localhost:5000` in your web browser while the backend Flask server is running (it is served automatically!). Alternatively, you can open `apps/web-dashboard/index.html` directly in your browser.

#### B. Mobile Application (`apps/mobile-app`)
* Rich client mobile application built using React Native and Expo, featuring a video scroll feed and manual scanner.
* Setup & Run:
  ```bash
  cd apps/mobile-app
  
  # Install node package dependencies
  npm install
  
  # Start the Expo development server
  npm start
  ```

---

## ⚠️ Production Constraints & Engineering Nuances

To ensure transparency and highlight optimization vectors, the current iteration of TrueFluence acknowledges the following constraints:

* **Generalization Variance**: The visual and audio components were base-trained on a localized dataset. Inference on out-of-distribution environments may experience variance, presenting an opportunity for scaled distributed training.
* **Graceful Degradation**: The Comments Engine enforces a strict dependency on the `transformers` library. Should the environment fail to instantiate it, the pipeline silently falls back to a neutral `0.5` engagement score to maintain system availability.