# TrueFluence — Multimodal Scam Detection System

An AI system that combines visual and audio analysis to detect scam videos using temporal analysis, voice feature extraction, and a phased fusion training strategy.

---

## 🚀 Key Features

### Visual Analysis
- **MobileNetV2 Backbone**: Frozen ImageNet-pretrained feature extraction (1280-dim)
- **Quality Assessment Head**: Multi-layer MLP scoring production quality per frame
- **Temporal LSTM**: Bidirectional 2-layer LSTM (hidden=256) over frame sequences
- **Temporal Self-Attention**: Learns which frames are most important (visual only)
- **Temporal Classifier**: Final LSTM → score pipeline

### Audio Analysis
- **128-dim Feature Vector**: MFCCs (80) + Chroma (24) + Spectral (6) + RMS (2) + Tempo (1) + Mel Bands (15)
- **Pause Pattern Analysis**: Voice authenticity via ffmpeg → wav → librosa pipeline

### Fusion
- **MLP Fusion Network**: Concatenates visual (135-dim) + audio (135-dim) → 270-dim → dense layers → final score
- **No-Audio Fallback**: Videos without audio use `audio_vector = zeros(130)` → score = 0.0
- **Fusion is concatenation-based MLP**

### Attention Clarification
| Component | Type | Input |
|---|---|---|
| `temporal_attention` | Self-attention over time | Visual frames only |
| `fusion_network` | MLP (concatenation) | Visual + Audio |

---

## 📁 Project Structure

```
Multimodals/
├── visual_engine.py          # MobileNetV2 backbone, LSTM, attention, fusion
├── audio_engine.py           # AudioFeatureExtractor + AdvancedAudioAnalyzer
├── train.py                  # 4-phase sequential training orchestrator
├── test.py                   # Inference pipeline + report generation
├── requirements.txt          # All dependencies
├── setup_project.py          # Project initialization
├── dataset/
│   ├── real_videos/          # Legitimate video samples (label=1)
│   ├── scam_videos/          # Scam video samples     (label=0)
│   └── processed_frames/     # Cached frame extractions
└── models/
    └── weights/
        ├── best_visual_head.pth      # Phase 1 best
        ├── best_visual_temporal.pth  # Phase 2 best
        ├── best_audio_head.pth       # Phase 3 best
        ├── best_fusion.pth           # Phase 4 best
        ├── best_model.pth            # Overall best (all components)
        └── final_model.pth           # Final epoch snapshot
```

---

## 🛠️ Installation

```bash
cd Multimodals
pip install -r requirements.txt
python setup_project.py
```

---

## 📦 Requirements

```
torch, torchvision, torchaudio
opencv-python
scikit-learn
pandas
numpy
librosa
scipy
tqdm
imageio-ffmpeg
resampy
soundfile
audioread
numba
```

> ⚠️ `moviepy` is listed in `requirements.txt` but is **not used** in the current pipeline. All audio extraction uses `imageio-ffmpeg` directly.

---

## 🎯 Usage

### Training

```bash
python train.py
```

Place videos before running:
- Real videos → `dataset/real_videos/`
- Scam videos → `dataset/scam_videos/`

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

### Testing

```bash
python test.py
```

Results saved to:
- `Test_Dataset/results.txt`
- `Test_Dataset/results.json`

### Programmatic

```python
from visual_engine import VisualQualityHead
from audio_engine  import AdvancedAudioAnalyzer
from train         import AudioClassificationHead
import torch

device       = torch.device('cpu')
visual_model = VisualQualityHead().to(device)
audio_head   = AudioClassificationHead(dropout=0.3).to(device)

ckpt = torch.load('models/weights/best_model.pth', map_location=device, weights_only=True)
visual_model.head.load_state_dict(ckpt['head'])
visual_model.temporal_lstm.load_state_dict(ckpt['temporal_lstm'])
visual_model.temporal_attention.load_state_dict(ckpt['temporal_attention'])
visual_model.temporal_classifier.load_state_dict(ckpt['temporal_classifier'])
visual_model.fusion_network.load_state_dict(ckpt['fusion_network'])
audio_head.load_state_dict(ckpt['audio_head'])
```

---

## 🧠 Training Architecture — 4-Phase Sequential

Each phase freezes all previously trained components and trains only the current target. This prevents catastrophic forgetting on small datasets.

```
Phase 1 — Visual Quality Head
  Trains  : model.head
  Freezes : backbone, temporal_lstm, temporal_attention,
            temporal_classifier, fusion_network
  Input   : (N, 1280) backbone vectors → mean logit
  Loss    : BCEWithLogitsLoss(pos_weight)

Phase 2 — Visual Temporal (LSTM + Attention)
  Trains  : temporal_lstm, temporal_attention, temporal_classifier
  Freezes : backbone, head (Phase 1), fusion_network
  Input   : (1, N, 3, 224, 224) frame sequence
  Loss    : BCEWithLogitsLoss(pos_weight)

Phase 3 — Audio Classification Head
  Trains  : AudioClassificationHead (128 → 64 → 1)
  Freezes : ALL visual components, VGGish (always frozen)
  Input   : 128-dim audio feature vector
  Fallback: No audio → score = 0.0, skip gradient update
  Loss    : BCEWithLogitsLoss(pos_weight)

Phase 4 — Fusion Network
  Trains  : model.fusion_network
  Freezes : ALL previous components
  Input   : visual_vec (135-dim) + audio_padded (135-dim)
  Loss    : BCEWithLogitsLoss(pos_weight)
```

### Key Training Decisions

| Setting | Value | Reason |
|---|---|---|
| `max_epochs_per_phase` | 5 | Small dataset (~15 videos) |
| `early_stop_patience` | 5 | More chances on tiny val set |
| `val_split` | 0.2 | Stratified 80/20 |
| `audio_dropout` | 0.5 | Reduce overfitting |
| `no_audio_score` | 0.0 | Conservative: no audio = suspicious |
| `pos_weight` | `num_real / num_scam` | Auto-adapts to class imbalance |

---

## 📊 Verdict Zones

| Score Range | Verdict | Emoji |
|---|---|---|
| 0.0 – 0.3 | SCAM | 🔴 |
| 0.3 – 0.5 | LIKELY SCAM | 🟠 |
| 0.5 – 0.7 | UNCERTAIN | 🟡 |
| 0.7 – 1.0 | REAL | 🟢 |

---

## 📋 Example Output

```
VIDEO 1: test_vid1.mp4
-----------------------------------------------------------------
  Visual Head Score    : 0.5005
  Temporal LSTM Score  : 0.5331

  Frame Scores:
    Frame 01           : 0.4636
    Frame 07           : 0.8450
    Frame 08           : 0.7128

  Has Audio            : True
  Audio Head Score     : 0.5419
  Pause Pattern Score  : 0.5867
  Consistency Score    : 0.5893
  Fusion Score         : 0.5138

  FINAL SCORE          : 0.5206
  VERDICT              : 🟡 UNCERTAIN (Borderline)
  Processing Time      : 14.75s
```

---

## ⚠️ Known Limitations

- Trained on small dataset (~15 videos) — generalization is limited
- Cross-modal attention is **not implemented** due to low system resources
- Fusion is using MLP concatenation
- No data augmentation currently applied during training