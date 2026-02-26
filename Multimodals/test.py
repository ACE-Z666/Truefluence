import os
import sys
import time
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import librosa
from datetime import datetime

from visual_engine import VisualQualityHead, extract_quality_frames
from audio_engine  import AdvancedAudioAnalyzer
from train         import AudioClassificationHead, extract_visual_features, extract_audio_features

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    'test_dir'          : os.path.join('Test_Dataset'),
    'weights_path'      : os.path.join('models', 'weights', 'best_model.pth'),
    'results_txt'       : os.path.join('Test_Dataset', 'results.txt'),
    'results_json'      : os.path.join('Test_Dataset', 'results.json'),

    # Video formats to scan
    'video_formats'     : ['*.mp4', '*.avi', '*.mov', '*.mkv'],

    # Frame extraction
    'num_frames'        : 8,

    # Audio head
    'audio_dropout'     : 0.3,

    # Verdict thresholds (Confidence Zones)
    'thresholds': {
        'scam'          : 0.3,      # 0.0 - 0.3  → SCAM
        'likely_scam'   : 0.5,      # 0.3 - 0.5  → LIKELY SCAM
        'uncertain'     : 0.7,      # 0.5 - 0.7  → UNCERTAIN
                                    # 0.7 - 1.0  → REAL
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_verdict(score):
    """
    Map score → confidence zone verdict.

    Zones:
        0.0 - 0.3  → 🔴 SCAM          (confident fake)
        0.3 - 0.5  → 🟠 LIKELY SCAM   (suspicious)
        0.5 - 0.7  → 🟡 UNCERTAIN     (borderline)
        0.7 - 1.0  → 🟢 REAL          (confident real)

    Returns:
        emoji  : str
        label  : str
        zone   : str
    """
    if score <= CONFIG['thresholds']['scam']:
        return '🔴', 'SCAM',        'Confident Fake'
    elif score <= CONFIG['thresholds']['likely_scam']:
        return '🟠', 'LIKELY SCAM', 'Suspicious'
    elif score <= CONFIG['thresholds']['uncertain']:
        return '🟡', 'UNCERTAIN',   'Borderline'
    else:
        return '🟢', 'REAL',        'Confident Real'


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_models(device):
    """
    Load all trained components from best_model.pth.

    Components loaded:
        visual_model.head
        visual_model.temporal_lstm
        visual_model.temporal_attention
        visual_model.temporal_classifier
        visual_model.fusion_network
        audio_head

    Returns:
        visual_model : VisualQualityHead
        audio_head   : AudioClassificationHead
        audio_analyzer: AdvancedAudioAnalyzer
    """
    print(f"\n  Loading weights from:")
    print(f"  {CONFIG['weights_path']}")

    if not os.path.exists(CONFIG['weights_path']):
        print(f"\n  ❌ Weights not found: {CONFIG['weights_path']}")
        print(f"     Run train.py first to generate weights.")
        sys.exit(1)

    # Load checkpoint
    ckpt = torch.load(
        CONFIG['weights_path'],
        map_location = device,
        weights_only = True
    )

    # Visual model
    visual_model = VisualQualityHead().to(device)
    visual_model.head.load_state_dict(ckpt['head'])
    visual_model.temporal_lstm.load_state_dict(ckpt['temporal_lstm'])
    visual_model.temporal_attention.load_state_dict(ckpt['temporal_attention'])
    visual_model.temporal_classifier.load_state_dict(ckpt['temporal_classifier'])
    visual_model.fusion_network.load_state_dict(ckpt['fusion_network'])
    visual_model.eval()

    # Audio head
    audio_head = AudioClassificationHead(
        dropout = CONFIG['audio_dropout']
    ).to(device)
    audio_head.load_state_dict(ckpt['audio_head'])
    audio_head.eval()

    # Audio analyzer (VGGish always frozen)
    audio_analyzer = AdvancedAudioAnalyzer(device=str(device))

    print(f"  ✅ All components loaded successfully")
    print(f"     Checkpoint epoch : {ckpt.get('epoch',  'N/A')}")
    print(f"     Best val_loss    : {ckpt.get('val_loss','N/A')}")
    print(f"     Best val_acc     : {ckpt.get('val_acc', 'N/A')}")

    return visual_model, audio_head, audio_analyzer


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE SINGLE VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video(video_path, visual_model, audio_head, audio_analyzer, device):
    """
    Full analysis pipeline for a single video.

    Steps:
        1. Extract frames → visual quality scores (per frame)
        2. Extract frames → temporal LSTM score
        3. Extract audio  → VGGish → audio head score
        4. Extract audio  → pause pattern score
        5. Extract audio  → consistency score
        6. Fusion         → final combined score

    Returns:
        dict with all scores and metadata
    """
    result = {
        'video_path'        : video_path,
        'video_name'        : os.path.basename(video_path),
        'has_audio'         : False,
        'frame_scores'      : [],
        'visual_head_score' : 0.0,
        'temporal_score'    : 0.0,
        'audio_head_score'  : 0.0,
        'pause_score'       : 0.0,
        'consistency_score' : 0.0,
        'fusion_score'      : 0.0,
        'final_score'       : 0.0,
        'processing_time'   : 0.0,
        'error'             : None,
    }

    t_start = time.time()

    try:
        with torch.no_grad():

            # ── VISUAL: FRAME QUALITY SCORES ─────────────────────────────
            frames, frames_batch, frame_vecs = extract_visual_features(
                video_path, visual_model, device
            )
            # frame_vecs: (num_frames, 1280)

            # Per-frame quality score through head
            raw_logits   = visual_model.head[:-1](frame_vecs)   # (num_frames, 1)
            frame_scores = torch.sigmoid(raw_logits).squeeze(1) # (num_frames,)
            frame_scores_list = frame_scores.cpu().numpy().tolist()

            # Average visual head score
            avg_logit         = raw_logits.mean().unsqueeze(0)
            visual_head_score = torch.sigmoid(avg_logit).item()

            result['frame_scores']       = [round(s, 4) for s in frame_scores_list]
            result['visual_head_score']  = round(visual_head_score, 4)

            # ── VISUAL: TEMPORAL LSTM SCORE ───────────────────────────────
            temporal_out   = visual_model.forward_temporal(frames_batch)  # (1,1)
            temporal_score = temporal_out.item()
            result['temporal_score'] = round(temporal_score, 4)

            # ── AUDIO FEATURES ────────────────────────────────────────────
            vggish_emb, audio_vector, has_audio = extract_audio_features(
                video_path, audio_analyzer, device
            )
            result['has_audio'] = has_audio

            if has_audio:
                # Audio head score
                audio_logit      = audio_head(vggish_emb)          # (1, 1)
                audio_head_score = torch.sigmoid(audio_logit).item()
                result['audio_head_score'] = round(audio_head_score, 4)

                # Pause score (index 128 of audio_vector)
                pause_score = audio_vector[128].item()
                result['pause_score'] = round(pause_score, 4)

                # Consistency score (index 129 of audio_vector)
                consistency_score = audio_vector[129].item()
                result['consistency_score'] = round(consistency_score, 4)

            else:
                # No audio → all audio scores = 0.0
                result['audio_head_score']  = 0.0
                result['pause_score']       = 0.0
                result['consistency_score'] = 0.0

            # ── FUSION SCORE ──────────────────────────────────────────────
            visual_vec            = frame_vecs.mean(dim=0)[:135].unsqueeze(0)
            audio_padded          = torch.zeros(1, 135, device=device)
            audio_padded[0, :130] = audio_vector

            fusion_out   = visual_model.forward_fusion(visual_vec, audio_padded)
            fusion_score = fusion_out.item()
            result['fusion_score'] = round(fusion_score, 4)

            # ── FINAL SCORE ───────────────────────────────────────────────
            # Weighted average of all scores
            # Fusion carries most weight (trained on all phases)
            if has_audio:
                final_score = (
                    0.20 * visual_head_score    +   # Phase 1
                    0.20 * temporal_score       +   # Phase 2
                    0.20 * audio_head_score     +   # Phase 3
                    0.40 * fusion_score             # Phase 4 (highest weight)
                )
            else:
                # No audio → visual only
                final_score = (
                    0.40 * visual_head_score    +   # Phase 1 (more weight)
                    0.60 * temporal_score           # Phase 2 (most weight)
                )

            result['final_score'] = round(final_score, 4)

    except Exception as e:
        result['error'] = str(e)
        print(f"  ⚠  Error analyzing {os.path.basename(video_path)}: {e}")

    result['processing_time'] = round(time.time() - t_start, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PRINT DETAILED REPORT (Terminal)
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result, index, total):
    """
    Print detailed analysis report for one video to terminal.
    """
    name          = result['video_name']
    final_score   = result['final_score']
    emoji, label, zone = get_verdict(final_score)

    print("\n" + "═" * 65)
    print(f"  VIDEO {index}/{total} : {name}")
    print("═" * 65)

    if result['error']:
        print(f"  ❌ Error: {result['error']}")
        return

    # ── VISUAL SCORES ─────────────────────────────────────────────────────
    print(f"\n  ── VISUAL ANALYSIS ──────────────────────────────────────")

    v_emoji, v_label, _ = get_verdict(result['visual_head_score'])
    print(f"  Quality Head Score  : {result['visual_head_score']:.4f}  {v_emoji} {v_label}")

    t_emoji, t_label, _ = get_verdict(result['temporal_score'])
    print(f"  Temporal LSTM Score : {result['temporal_score']:.4f}  {t_emoji} {t_label}")

    print(f"\n  Frame-by-Frame Quality Scores:")
    for i, score in enumerate(result['frame_scores']):
        f_emoji, f_label, _ = get_verdict(score)
        bar_len  = int(score * 30)
        bar      = '█' * bar_len + '░' * (30 - bar_len)
        print(f"    Frame {i+1:02d} : [{bar}] {score:.4f}  {f_emoji}")

    # ── AUDIO SCORES ──────────────────────────────────────────────────────
    print(f"\n  ── AUDIO ANALYSIS ───────────────────────────────────────")

    if result['has_audio']:
        a_emoji, a_label, _ = get_verdict(result['audio_head_score'])
        p_emoji, p_label, _ = get_verdict(result['pause_score'])
        c_emoji, c_label, _ = get_verdict(result['consistency_score'])

        print(f"  Audio Head Score    : {result['audio_head_score']:.4f}  {a_emoji} {a_label}")
        print(f"  Pause Pattern Score : {result['pause_score']:.4f}  {p_emoji} {p_label}")
        print(f"  Consistency Score   : {result['consistency_score']:.4f}  {c_emoji} {c_label}")
    else:
        print(f"  ⚠  No audio track detected")
        print(f"  Audio scores set to : 0.0000  (fallback)")

    # ── FUSION SCORE ──────────────────────────────────────────────────────
    print(f"\n  ── FUSION ANALYSIS ──────────────────────────────────────")
    fu_emoji, fu_label, _ = get_verdict(result['fusion_score'])
    print(f"  Fusion Score        : {result['fusion_score']:.4f}  {fu_emoji} {fu_label}")

    # ── SCORE BREAKDOWN ───────────────────────────────────────────────────
    print(f"\n  ── SCORE BREAKDOWN ──────────────────────────────────────")

    if result['has_audio']:
        print(f"  Visual Head  (20%)  : {result['visual_head_score']:.4f} × 0.20 "
              f"= {result['visual_head_score'] * 0.20:.4f}")
        print(f"  Temporal     (20%)  : {result['temporal_score']:.4f} × 0.20 "
              f"= {result['temporal_score'] * 0.20:.4f}")
        print(f"  Audio Head   (20%)  : {result['audio_head_score']:.4f} × 0.20 "
              f"= {result['audio_head_score'] * 0.20:.4f}")
        print(f"  Fusion       (40%)  : {result['fusion_score']:.4f} × 0.40 "
              f"= {result['fusion_score'] * 0.40:.4f}")
    else:
        print(f"  Visual Head  (40%)  : {result['visual_head_score']:.4f} × 0.40 "
              f"= {result['visual_head_score'] * 0.40:.4f}")
        print(f"  Temporal     (60%)  : {result['temporal_score']:.4f} × 0.60 "
              f"= {result['temporal_score'] * 0.60:.4f}")
        print(f"  Audio        (  %)  : N/A (no audio)")

    # ── FINAL VERDICT ─────────────────────────────────────────────────────
    print(f"\n  {'─' * 55}")
    print(f"  FINAL SCORE         : {final_score:.4f}")
    print(f"  VERDICT             : {emoji}  {label}  ({zone})")
    print(f"  Processing Time     : {result['processing_time']}s")
    print(f"  {'─' * 55}")

    # Confidence bar
    bar_len = int(final_score * 50)
    bar     = '█' * bar_len + '░' * (50 - bar_len)
    print(f"\n  Confidence: [{bar}]")
    print(f"              SCAM ◄──────────────────────────► REAL")
    print(f"              0.0   0.3   0.5   0.7   1.0")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE REPORT (File)
# ─────────────────────────────────────────────────────────────────────────────

def save_report(all_results):
    """
    Save detailed results to:
        Test_Dataset/results.txt  (human readable)
        Test_Dataset/results.json (machine readable)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── TXT REPORT ────────────────────────────────────────────────────────
    txt_lines = []
    txt_lines.append("=" * 65)
    txt_lines.append("  TRUEFLUENCE — MULTIMODAL SCAM DETECTION")
    txt_lines.append("  Test Report")
    txt_lines.append(f"  Generated : {timestamp}")
    txt_lines.append("=" * 65)

    for i, result in enumerate(all_results, 1):
        final_score        = result['final_score']
        emoji, label, zone = get_verdict(final_score)

        txt_lines.append(f"\nVIDEO {i}: {result['video_name']}")
        txt_lines.append("-" * 65)

        if result['error']:
            txt_lines.append(f"  ERROR: {result['error']}")
            continue

        txt_lines.append(f"  Visual Head Score    : {result['visual_head_score']:.4f}")
        txt_lines.append(f"  Temporal LSTM Score  : {result['temporal_score']:.4f}")

        txt_lines.append(f"\n  Frame Scores:")
        for j, s in enumerate(result['frame_scores']):
            txt_lines.append(f"    Frame {j+1:02d}          : {s:.4f}")

        txt_lines.append(f"\n  Has Audio            : {result['has_audio']}")
        txt_lines.append(f"  Audio Head Score     : {result['audio_head_score']:.4f}")
        txt_lines.append(f"  Pause Pattern Score  : {result['pause_score']:.4f}")
        txt_lines.append(f"  Consistency Score    : {result['consistency_score']:.4f}")
        txt_lines.append(f"  Fusion Score         : {result['fusion_score']:.4f}")

        txt_lines.append(f"\n  FINAL SCORE          : {final_score:.4f}")
        txt_lines.append(f"  VERDICT              : {label} ({zone})")
        txt_lines.append(f"  Processing Time      : {result['processing_time']}s")

    # Summary
    txt_lines.append("\n" + "=" * 65)
    txt_lines.append("  SUMMARY")
    txt_lines.append("=" * 65)
    for result in all_results:
        if not result['error']:
            emoji, label, zone = get_verdict(result['final_score'])
            txt_lines.append(
                f"  {result['video_name']:<35} "
                f"{result['final_score']:.4f}  {emoji} {label}"
            )

    txt_content = "\n".join(txt_lines)

    # Write TXT
    with open(CONFIG['results_txt'], 'w', encoding='utf-8') as f:
        f.write(txt_content)

    # ── JSON REPORT ───────────────────────────────────────────────────────
    json_data = {
        'timestamp'     : timestamp,
        'weights_used'  : CONFIG['weights_path'],
        'total_videos'  : len(all_results),
        'results'       : []
    }

    for result in all_results:
        if not result['error']:
            emoji, label, zone = get_verdict(result['final_score'])
            json_data['results'].append({
                **result,
                'verdict'       : label,
                'verdict_zone'  : zone,
                'verdict_emoji' : emoji,
            })
        else:
            json_data['results'].append(result)

    with open(CONFIG['results_json'], 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print(f"\n  ✅ Report saved:")
    print(f"     TXT  → {CONFIG['results_txt']}")
    print(f"     JSON → {CONFIG['results_json']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def test():
    print("\n" + "═" * 65)
    print("  🔍 TRUEFLUENCE — MULTIMODAL SCAM DETECTION TEST")
    print("     Visual  : MobileNetV2 + LSTM + Attention")
    print("     Audio   : VGGish (frozen) + Pause + Audio Head")
    print("     Verdict : Confidence Zone Thresholds")
    print("═" * 65)

    total_start = time.time()

    # ── DEVICE ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")

    # ── COLLECT TEST VIDEOS ──────────────────────────────────────────────────
    print(f"\n  Scanning : {CONFIG['test_dir']}")

    if not os.path.exists(CONFIG['test_dir']):
        print(f"  ❌ Test directory not found: {CONFIG['test_dir']}")
        sys.exit(1)

    test_videos = []
    for fmt in CONFIG['video_formats']:
        test_videos += glob.glob(os.path.join(CONFIG['test_dir'], fmt))

    if len(test_videos) == 0:
        print(f"  ❌ No videos found in {CONFIG['test_dir']}")
        sys.exit(1)

    test_videos.sort()
    print(f"  Found {len(test_videos)} video(s):")
    for v in test_videos:
        print(f"    → {os.path.basename(v)}")

    # ── LOAD MODELS ──────────────────────────────────────────────────────────
    print(f"\n  Loading Models...")
    visual_model, audio_head, audio_analyzer = load_models(device)

    # ── ANALYZE EACH VIDEO ───────────────────────────────────────────────────
    print(f"\n  Analyzing {len(test_videos)} video(s)...")

    all_results = []

    for i, video_path in enumerate(test_videos, 1):
        result = analyze_video(
            video_path,
            visual_model,
            audio_head,
            audio_analyzer,
            device
        )
        all_results.append(result)
        print_report(result, i, len(test_videos))

    # ── OVERALL SUMMARY ──────────────────────────────────────────────────────
    total_time = time.time() - total_start

    print("\n" + "═" * 65)
    print("  OVERALL SUMMARY")
    print("═" * 65)
    print(f"  {'Video':<35} {'Score':>6}  Verdict")
    print("  " + "-" * 55)

    for result in all_results:
        if not result['error']:
            emoji, label, zone = get_verdict(result['final_score'])
            print(f"  {result['video_name']:<35} "
                  f"{result['final_score']:>6.4f}  "
                  f"{emoji} {label}")
        else:
            print(f"  {result['video_name']:<35}  ❌ ERROR")

    print("  " + "-" * 55)
    print(f"  Total Processing Time : {total_time:.2f}s")
    print("═" * 65)

    # ── SAVE REPORT ──────────────────────────────────────────────────────────
    save_report(all_results)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test()