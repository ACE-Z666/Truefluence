"""
evaluate_metrics.py
────────────────────────────────────────────────────────────────────────────
TrueFluence — Evaluation Metrics
Runs the full sequential pipeline on ALL labeled videos
(dataset/real_videos/ + dataset/scam_videos/) and prints:

  • Accuracy
  • AUC-ROC
  • F1 Score  (macro)
  • Precision (macro)
  • Recall    (macro)
  • Confusion Matrix

Ground-truth labels: real_videos/ → 1  (REAL),  scam_videos/ → 0  (SCAM)
Positive class     : REAL  (label = 1)
Classification threshold: final_score > 0.5 → predicted REAL

Run from:  d:\\Truefluence\\Multimodals\\
Usage   :  python evaluate_metrics.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import glob
import time

import numpy as np
import torch

# ── sklearn metrics ──────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── TrueFluence pipeline ─────────────────────────────────────────────────────
from visual_engine  import VisualQualityHead
from audio_engine   import AdvancedAudioAnalyzer
from mesonet        import load_meso4, screen_for_deepfake
from train          import (
    AudioClassificationHead,
    extract_visual_features,
    extract_audio_features,
    CONFIG as TRAIN_CFG,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EVAL_CFG = {
    'real_dir'          : os.path.join('dataset', 'real_videos'),
    'scam_dir'          : os.path.join('dataset', 'scam_videos'),
    'weights_path'      : os.path.join('models', 'weights', 'best_model.pth'),
    'meso_weights'      : os.path.join('models', 'weights', 'meso4_DF.pth'),

    # Same hyperparams as test.py
    'num_frames'        : 8,
    'audio_dropout'     : 0.3,
    'deepfake_threshold': 0.80,
    'no_audio_score'    : 0.0,

    # Same score-fusion weights as test.py
    'w_video_audio'     : 0.40,
    'w_comments_eng'    : 0.60,

    # Verdict threshold (score > 0.5 → REAL)
    'classify_threshold': 0.50,

    # Video formats
    'video_formats'     : ['*.mp4', '*.avi', '*.mov', '*.mkv'],
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS  (identical to test.py load_models)
# ─────────────────────────────────────────────────────────────────────────────

def load_models(device):
    print(f"\n  Loading TrueFluence weights …")

    if not os.path.exists(EVAL_CFG['weights_path']):
        print(f"  ❌ Weights not found: {EVAL_CFG['weights_path']}")
        print(f"     Run train.py first.")
        sys.exit(1)

    ckpt = torch.load(
        EVAL_CFG['weights_path'],
        map_location = device,
        weights_only = True
    )

    visual_model = VisualQualityHead().to(device)
    visual_model.head.load_state_dict(ckpt['head'])
    visual_model.temporal_lstm.load_state_dict(ckpt['temporal_lstm'])
    visual_model.temporal_attention.load_state_dict(ckpt['temporal_attention'])
    visual_model.temporal_classifier.load_state_dict(ckpt['temporal_classifier'])
    visual_model.fusion_network.load_state_dict(ckpt['fusion_network'])
    visual_model.eval()

    audio_head = AudioClassificationHead(dropout=EVAL_CFG['audio_dropout']).to(device)
    audio_head.load_state_dict(ckpt['audio_head'])
    audio_head.eval()

    audio_analyzer = AdvancedAudioAnalyzer(device=str(device))

    print(f"  ✅ TrueFluence weights loaded  (epoch {ckpt.get('epoch','N/A')})")

    meso_model = load_meso4(device, EVAL_CFG['meso_weights'])

    return visual_model, audio_head, audio_analyzer, meso_model


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT LABELED VIDEOS
# ─────────────────────────────────────────────────────────────────────────────

def collect_labeled_videos():
    videos, labels = [], []

    for fmt in EVAL_CFG['video_formats']:
        for v in glob.glob(os.path.join(EVAL_CFG['real_dir'], fmt)):
            videos.append(v)
            labels.append(1)   # REAL

    for fmt in EVAL_CFG['video_formats']:
        for v in glob.glob(os.path.join(EVAL_CFG['scam_dir'], fmt)):
            videos.append(v)
            labels.append(0)   # SCAM

    print(f"  Real  videos : {sum(1 for l in labels if l == 1)}")
    print(f"  Scam  videos : {sum(1 for l in labels if l == 0)}")
    print(f"  Total        : {len(videos)}")
    return videos, labels


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ONE VIDEO  (mirrors test.py analyze_video, minus comments engine)
# ─────────────────────────────────────────────────────────────────────────────

def score_video(video_path, visual_model, audio_head, audio_analyzer, meso_model, device):
    """
    Returns (final_score, raw_video_audio_score).
    Comments+Engagement uses a neutral 0.5 fallback (no real data for training vids).
    """
    try:
        # ── MesoNet gate ──────────────────────────────────────────────────────
        meso_result = screen_for_deepfake(
            video_path, meso_model, device,
            num_frames = 16,
            threshold  = EVAL_CFG['deepfake_threshold'],
        )
        if meso_result['is_deepfake']:
            return 0.0, 0.0     # Deepfake → SCAM score

        with torch.no_grad():

            # ── Visual features ───────────────────────────────────────────────
            frames, frames_batch, frame_vecs = extract_visual_features(
                video_path, visual_model, device
            )

            raw_logits        = visual_model.head[:-1](frame_vecs)
            avg_logit         = raw_logits.mean().unsqueeze(0)
            visual_head_score = torch.sigmoid(avg_logit).item()

            temporal_out   = visual_model.forward_temporal(frames_batch)
            temporal_score = temporal_out.item()

            # ── Audio features ────────────────────────────────────────────────
            vggish_emb, audio_vector, has_audio = extract_audio_features(
                video_path, audio_analyzer, device
            )

            if has_audio:
                audio_logit      = audio_head(vggish_emb)
                audio_head_score = torch.sigmoid(audio_logit).item()
            else:
                audio_head_score = EVAL_CFG['no_audio_score']

            # ── Fusion ────────────────────────────────────────────────────────
            visual_vec            = frame_vecs.mean(dim=0)[:135].unsqueeze(0)
            audio_padded          = torch.zeros(1, 135, device=device)
            audio_padded[0, :130] = audio_vector
            fusion_out   = visual_model.forward_fusion(visual_vec, audio_padded)
            fusion_score = fusion_out.item()

            # ── Video+Audio blended score (40 % bucket) ───────────────────────
            if has_audio:
                video_audio_score = (
                    0.15 * visual_head_score +
                    0.15 * temporal_score    +
                    0.15 * audio_head_score  +
                    0.55 * fusion_score
                )
            else:
                video_audio_score = (
                    0.40 * visual_head_score +
                    0.60 * temporal_score
                )

        # ── Final: 40 % V+A  +  60 % Comments (neutral 0.5 fallback) ─────────
        comments_eng_score = 0.5
        final_score = (
            EVAL_CFG['w_video_audio']  * video_audio_score +
            EVAL_CFG['w_comments_eng'] * comments_eng_score
        )

        return round(final_score, 6), round(video_audio_score, 6)

    except Exception as e:
        print(f"  ⚠  Error scoring {os.path.basename(video_path)}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 70)
    print("  📊 TRUEFLUENCE — EVALUATION METRICS")
    print("  Dataset : dataset/real_videos/  +  dataset/scam_videos/")
    print("  Metrics : Accuracy · AUC-ROC · F1 · Precision · Recall")
    print("  Note    : Comments+Engagement uses neutral 0.5 fallback")
    print("            (no social data for training videos)")
    print("═" * 70)

    t0 = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")

    # ── Collect videos ───────────────────────────────────────────────────────
    print("\n[1] Collecting labeled videos …")
    videos, true_labels = collect_labeled_videos()

    if len(videos) == 0:
        print("  ❌ No videos found.")
        sys.exit(1)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[2] Loading models …")
    visual_model, audio_head, audio_analyzer, meso_model = load_models(device)

    # ── Score every video ────────────────────────────────────────────────────
    print(f"\n[3] Scoring {len(videos)} video(s) …\n")

    scores      = []   # continuous final_score  → for AUC-ROC
    pred_labels = []   # binary prediction       → for F1 / Accuracy
    valid_true  = []   # ground-truth for matched scored videos

    header = f"  {'Video':<35}  {'GT':>5}  {'Score':>7}  {'Pred':>10}"
    print(header)
    print("  " + "-" * 62)

    for video_path, gt_label in zip(videos, true_labels):
        name = os.path.basename(video_path)
        final_score, _ = score_video(
            video_path,
            visual_model, audio_head, audio_analyzer, meso_model,
            device
        )

        if final_score is None:
            print(f"  {name:<35}  {'ERROR':>5}")
            continue

        pred = 1 if final_score > EVAL_CFG['classify_threshold'] else 0
        gt_str   = "REAL" if gt_label == 1 else "SCAM"
        pred_str = "REAL" if pred     == 1 else "SCAM"
        match    = "✅" if pred == gt_label else "❌"

        print(f"  {name:<35}  {gt_str:>5}  {final_score:>7.4f}  {pred_str:>10}  {match}")

        scores.append(final_score)
        pred_labels.append(pred)
        valid_true.append(gt_label)

    # ── Compute metrics ──────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  📈 EVALUATION RESULTS")
    print("═" * 70)

    n = len(valid_true)
    if n == 0:
        print("  ❌ No videos successfully scored.")
        sys.exit(1)

    y_true   = np.array(valid_true)
    y_scores = np.array(scores)
    y_pred   = np.array(pred_labels)

    accuracy  = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='macro',    zero_division=0)

    # AUC-ROC requires both classes present
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_scores)
    else:
        auc_roc = float('nan')
        print("  ⚠  AUC-ROC undefined — only one class present in dataset.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n  Videos Evaluated  : {n}")
    print(f"  Threshold Used    : {EVAL_CFG['classify_threshold']} (score > threshold → REAL)\n")

    print(f"  ┌───────────────────────────────────┐")
    print(f"  │  Accuracy   : {accuracy:>7.4f}  ({accuracy*100:.1f}%)  │")
    print(f"  │  AUC-ROC    : {auc_roc:>7.4f}              │")
    print(f"  │  F1 Score   : {f1:>7.4f}  (macro)        │")
    print(f"  │  Precision  : {precision:>7.4f}  (macro)        │")
    print(f"  │  Recall     : {recall:>7.4f}  (macro)        │")
    print(f"  └───────────────────────────────────┘")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=True, cols=Pred):")
    print(f"                Pred SCAM   Pred REAL")
    print(f"  True SCAM   :  {cm[0,0]:>8}    {cm[0,1]:>8}")
    print(f"  True REAL   :  {cm[1,0]:>8}    {cm[1,1]:>8}")

    # Per-class report
    print(f"\n  Per-Class Report:")
    print(classification_report(
        y_true, y_pred,
        target_names = ['SCAM (0)', 'REAL (1)'],
        zero_division = 0
    ))

    elapsed = time.time() - t0
    print(f"  Total time : {elapsed:.1f}s")
    print("═" * 70)


if __name__ == '__main__':
    main()
