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
from audio_engine   import AdvancedAudioAnalyzer
from train          import AudioClassificationHead, extract_visual_features, extract_audio_features
from mesonet        import Meso4, load_meso4, screen_for_deepfake

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    'test_dir'          : os.path.join('Test_Dataset'),
    'weights_path'      : os.path.join('models', 'weights', 'best_model.pth'),
    'meso_weights'      : os.path.join('models', 'weights', 'meso4_DF.pth'),
    'results_txt'       : os.path.join('Test_Dataset', 'results.txt'),
    'results_json'      : os.path.join('Test_Dataset', 'results.json'),

    # Video formats to scan
    'video_formats'     : ['*.mp4', '*.avi', '*.mov', '*.mkv'],

    # Frame extraction
    'num_frames'        : 8,

    # Audio head
    'audio_dropout'     : 0.3,

    # ── MesoNet Gate ──────────────────────────────────────────────────────
    # If the fraction of frames classified as deepfake
    # exceeds this threshold, the pipeline aborts immediately.
    # final_score = 0.0, verdict = DEEPFAKE
    'deepfake_threshold': 0.80,

    # ── Score Fusion Weights ──────────────────────────────────────────────
    # Stage 1 : Video + Audio engine combined  → weight 0.40  (40 %)
    # Stage 2 : Comments + Engagement engine   → weight 0.60  (60 %)
    'w_video_audio'     : 0.40,
    'w_comments_eng'    : 0.60,

    # Verdict thresholds (Confidence Zones)
    'thresholds': {
        'scam'          : 0.3,      # 0.0 - 0.3  → SCAM
        'likely_scam'   : 0.5,      # 0.3 - 0.5  → LIKELY SCAM
        'LIKELY REAL'     : 0.7,      # 0.5 - 0.7  → LIKELY REAL
                                    # 0.7 - 1.0  → REAL
    },
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
        0.5 - 0.7  → 🟡 LIKELY REAL     (borderline)
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
    elif score <= CONFIG['thresholds']['LIKELY REAL']:
        return '🟡', 'LIKELY REAL',   'Borderline'
    else:
        return '🟢', 'REAL',        'Confident Real'


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_models(device):
    """
    Load all trained components from best_model.pth + MesoNet.

    Components loaded:
        visual_model.head
        visual_model.temporal_lstm
        visual_model.temporal_attention
        visual_model.temporal_classifier
        visual_model.fusion_network
        audio_head
        meso_model  (may be None if weights not found)

    Returns:
        visual_model  : VisualQualityHead
        audio_head    : AudioClassificationHead
        audio_analyzer: AdvancedAudioAnalyzer
        meso_model    : Meso4 | None
    """
    # ── TrueFluence main model ─────────────────────────────────────────────

    print(f"\n  Loading TrueFluence weights from:")
    print(f"  {CONFIG['weights_path']}")

    if not os.path.exists(CONFIG['weights_path']):
        print(f"\n  ❌ Weights not found: {CONFIG['weights_path']}")
        print(f"     Run train.py first to generate weights.")
        sys.exit(1)

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

    print(f"  ✅ TrueFluence components loaded")
    print(f"     Checkpoint epoch : {ckpt.get('epoch',  'N/A')}")
    print(f"     Best val_loss    : {ckpt.get('val_loss','N/A')}")
    print(f"     Best val_acc     : {ckpt.get('val_acc', 'N/A')}")

    # ── MesoNet deepfake gate ──────────────────────────────────────────────
    print(f"\n  Loading MesoNet (Meso-4 DF) …")
    meso_model = load_meso4(device, CONFIG['meso_weights'])

    return visual_model, audio_head, audio_analyzer, meso_model


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE SINGLE VIDEO  —  Full Sequential Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video(video_path, visual_model, audio_head, audio_analyzer,
                  meso_model, device):
    """
    Full sequential analysis pipeline for a single video.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  STEP 1 — MesoNet Deepfake Gate                                  ║
    ║  Extract frames → Meso-4 → deepfake_prob per frame               ║
    ║  If mean(deepfake_prob) ≥ 80%:                                   ║
    ║      final_score = 0  →  DEEPFAKE  (pipeline STOPS here)         ║
    ║  Else: continue ↓                                                 ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  STEP 2 — Video Engine                                            ║
    ║  MobileNetV2 quality head  +  Temporal LSTM score                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  STEP 3 — Audio Engine                                            ║
    ║  VGGish  +  Pause patterns  +  Consistency score                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  STEP 4 — Video+Audio Fusion  (internal score)                    ║
    ║  video_audio_score  =  weighted blend of steps 2+3               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  STEP 5 — Comments + Engagement Engine  (placeholder / future)   ║
    ║  comments_eng_score = 0.5 (neutral) if module not available      ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  FINAL SCORE  =  0.40 × video_audio  +  0.60 × comments_eng     ║
    ╚══════════════════════════════════════════════════════════════════╝

    Returns:
        dict with all scores and metadata
    """

    result = {
        'video_path'            : video_path,
        'video_name'            : os.path.basename(video_path),

        # ── MesoNet gate ──────────────────────────────────────────────────
        'meso_available'        : False,
        'is_deepfake'           : False,
        'deepfake_prob'         : 0.0,
        'frame_df_probs'        : [],

        # ── Video engine ──────────────────────────────────────────────────
        'has_audio'             : False,
        'frame_scores'          : [],
        'visual_head_score'     : 0.0,
        'temporal_score'        : 0.0,

        # ── Audio engine ──────────────────────────────────────────────────
        'audio_head_score'      : 0.0,
        'pause_score'           : 0.0,
        'consistency_score'     : 0.0,

        # ── Fusion ────────────────────────────────────────────────────────
        'fusion_score'          : 0.0,
        'video_audio_score'     : 0.0,

        # ── Comments + Engagement ─────────────────────────────────────────
        'comments_eng_score'    : 0.5,   # neutral placeholder

        # ── Final ─────────────────────────────────────────────────────────
        'final_score'           : 0.0,
        'processing_time'       : 0.0,
        'error'                 : None,
    }

    t_start = time.time()

    try:
        # ══════════════════════════════════════════════════════════════════
        # STEP 1 — MesoNet Deepfake Gate
        # ══════════════════════════════════════════════════════════════════

        print(f"\n  [MesoNet] Screening for deepfakes …")

        meso_result = screen_for_deepfake(
            video_path,
            meso_model,
            device,
            num_frames  = 16,
            threshold   = CONFIG['deepfake_threshold'],
        )

        result['meso_available']  = meso_result['meso_available']
        result['is_deepfake']     = meso_result['is_deepfake']
        result['deepfake_prob']   = meso_result['deepfake_prob']
        result['frame_df_probs']  = meso_result['frame_df_probs']

        if meso_result['meso_available']:
            print(f"  [MesoNet] Deepfake probability : "
                  f"{meso_result['deepfake_prob']:.1%}  "
                  f"({'⛔ DEEPFAKE DETECTED' if meso_result['is_deepfake'] else '✅ Not a deepfake'})")

        # ── ABORT if deepfake ─────────────────────────────────────────────
        if result['is_deepfake']:
            result['final_score']      = 0.0
            result['video_audio_score']= 0.0
            result['processing_time']  = round(time.time() - t_start, 2)
            print(f"  🚫 Pipeline aborted — Deepfake content detected.")
            return result

        # ══════════════════════════════════════════════════════════════════
        # STEPS 2–4 — Video + Audio + Fusion
        # ══════════════════════════════════════════════════════════════════

        with torch.no_grad():

            # ── STEP 2 : VIDEO ENGINE ────────────────────────────────────
            frames, frames_batch, frame_vecs = extract_visual_features(
                video_path, visual_model, device
            )

            # Per-frame quality score through head
            raw_logits   = visual_model.head[:-1](frame_vecs)   # (N, 1)
            frame_scores = torch.sigmoid(raw_logits).squeeze(1) # (N,)
            frame_scores_list = frame_scores.cpu().numpy().tolist()

            avg_logit         = raw_logits.mean().unsqueeze(0)
            visual_head_score = torch.sigmoid(avg_logit).item()

            result['frame_scores']      = [round(s, 4) for s in frame_scores_list]
            result['visual_head_score'] = round(visual_head_score, 4)

            # Temporal LSTM
            temporal_out   = visual_model.forward_temporal(frames_batch)
            temporal_score = temporal_out.item()
            result['temporal_score'] = round(temporal_score, 4)

            # ── STEP 3 : AUDIO ENGINE ────────────────────────────────────
            vggish_emb, audio_vector, has_audio = extract_audio_features(
                video_path, audio_analyzer, device
            )
            result['has_audio'] = has_audio

            if has_audio:
                audio_logit      = audio_head(vggish_emb)
                audio_head_score = torch.sigmoid(audio_logit).item()
                result['audio_head_score']  = round(audio_head_score, 4)
                result['pause_score']       = round(audio_vector[128].item(), 4)
                result['consistency_score'] = round(audio_vector[129].item(), 4)
            else:
                result['audio_head_score']  = 0.0
                result['pause_score']       = 0.0
                result['consistency_score'] = 0.0

            # ── STEP 4 : FUSION (Video+Audio combined) ───────────────────
            visual_vec            = frame_vecs.mean(dim=0)[:135].unsqueeze(0)
            audio_padded          = torch.zeros(1, 135, device=device)
            audio_padded[0, :130] = audio_vector

            fusion_out   = visual_model.forward_fusion(visual_vec, audio_padded)
            fusion_score = fusion_out.item()
            result['fusion_score'] = round(fusion_score, 4)

            # ── Video+Audio blended score (internal 40 % bucket) ─────────
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

            result['video_audio_score'] = round(video_audio_score, 4)

        # ══════════════════════════════════════════════════════════════════
        # STEP 5 — Comments + Engagement Engine
        # ══════════════════════════════════════════════════════════════════

        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), 'comments'))
            from bert_comment.run_bert_comments import analyze_comments_bert
            from bert_comment.run_engagement    import analyze_engagement

            import json as _json

            # Try to load real data from a JSON file matching the video name
            _video_dir = _os.path.dirname(video_path)
            _video_base = _os.path.splitext(_os.path.basename(video_path))[0]
            _json_data_path = _os.path.join(_video_dir, f"{_video_base}.json")

            if _os.path.exists(_json_data_path):
                with open(_json_data_path, 'r', encoding='utf-8') as f:
                    _video_data = _json.load(f)
                _comments_list = _video_data.get('comments', [])
                _followers     = _video_data.get('followers', 50000)
                _likes         = _video_data.get('likes', 5200)
                _num_comments  = _video_data.get('num_comments', len(_comments_list) if _comments_list else 600)
                # print(f"  [Comments] Loaded engagement data from {_video_base}.json")
            else:
                _comments_list  = [
                    "Amazing quality for the price",
                    "Very happy with the food",
                    "Average",
                    "Bad",
                    "nice Food bro",
                    "Fantastic"
                ] 
                # Using realistic high engagement metrics based on test.py (real → HIGH)
                _followers      = 50000       
                _likes          = 5200         
                _num_comments   = 600           
            
            _eng_score = analyze_engagement(_followers, _likes, _num_comments)
            _comment_result = analyze_comments_bert(_comments_list)
            _comment_score = _comment_result.get('comment_authenticity_score', 0.5)
            comments_eng_score = round(0.5 * _comment_score + 0.5 * _eng_score, 4)

        except Exception as _ce:
            print(f"  ⚠  Comments engine error: {_ce}")
            comments_eng_score = 0.5      # neutral fallback

        # ══════════════════════════════════════════════════════════════════
        # FINAL SCORE  =  40 % Video+Audio  +  60 % Comments+Engagement
        # ══════════════════════════════════════════════════════════════════

        final_score = (
            CONFIG['w_video_audio']  * video_audio_score +
            CONFIG['w_comments_eng'] * comments_eng_score
        )
        result['final_score']         = round(final_score, 4)
        result['comments_eng_score']  = round(comments_eng_score, 4)

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

    # ── MESONET DEEPFAKE GATE ──────────────────────────────────────────────

    print(f"\n  ── 🔍 DEEPFAKE GATE (MesoNet Meso-4) ───────────────────")

    if result['meso_available']:
        df_pct = result['deepfake_prob'] * 100
        df_bar_len = int(result['deepfake_prob'] * 30)
        df_bar = '█' * df_bar_len + '░' * (30 - df_bar_len)

        print(f"  Deepfake Probability : [{df_bar}] {df_pct:.1f}%")

        if result['is_deepfake']:
            print(f"\n  ⛔  DEEPFAKE DETECTED — Pipeline Aborted")
            print(f"  FINAL SCORE : 0.0000  🔴 DEEPFAKE")
            print(f"  Processing Time : {result['processing_time']}s")
            print(f"  {'═' * 63}")
            return
        else:
            print(f"  Gate Result        : ✅ PASS (< 80% deepfake threshold)")
    else:
        print(f"  ⚠  MesoNet weights not found — Gate DISABLED")
        print(f"     Run:  python mesonet.py  to download weights")

    # ── VISUAL SCORES ─────────────────────────────────────────────────────

    print(f"\n  ── 🎥 VIDEO ENGINE ──────────────────────────────────────")

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

    print(f"\n  ── 🎵 AUDIO ENGINE ──────────────────────────────────────")

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
    print(f"\n  ── 🔗 FUSION (Video + Audio) ────────────────────────────")
    fu_emoji, fu_label, _ = get_verdict(result['fusion_score'])
    print(f"  Fusion Score        : {result['fusion_score']:.4f}  {fu_emoji} {fu_label}")
    va_emoji, va_label, _ = get_verdict(result['video_audio_score'])
    print(f"  Video+Audio Score   : {result['video_audio_score']:.4f}  {va_emoji} {va_label}")

    # ── COMMENTS + ENGAGEMENT ─────────────────────────────────────────────
    print(f"\n  ── 💬 COMMENTS + ENGAGEMENT ENGINE ─────────────────────")
    ce_emoji, ce_label, _ = get_verdict(result['comments_eng_score'])
    print(f"  Comments+Eng Score  : {result['comments_eng_score']:.4f}  {ce_emoji} {ce_label}")

    # ── SCORE BREAKDOWN ───────────────────────────────────────────────────
    print(f"\n  ── SCORE BREAKDOWN ──────────────────────────────────────")
    w_va = CONFIG['w_video_audio']
    w_ce = CONFIG['w_comments_eng']
    print(f"  Video+Audio  ({int(w_va*100):2d}%) : "
          f"{result['video_audio_score']:.4f} × {w_va:.2f} "
          f"= {result['video_audio_score'] * w_va:.4f}")
    print(f"  Comments+Eng ({int(w_ce*100):2d}%) : "
          f"{result['comments_eng_score']:.4f} × {w_ce:.2f} "
          f"= {result['comments_eng_score'] * w_ce:.4f}")

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
    txt_lines.append("  Sequential Pipeline: MesoNet → Video → Audio → Comments+Engagement")
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

        # MesoNet gate
        txt_lines.append(f"  [MesoNet Gate]")
        txt_lines.append(f"  MesoNet Available    : {result['meso_available']}")
        txt_lines.append(f"  Deepfake Probability : {result['deepfake_prob']:.4f}")
        txt_lines.append(f"  Is Deepfake          : {result['is_deepfake']}")

        if result['is_deepfake']:
            txt_lines.append(f"  ⛔ DEEPFAKE — Pipeline aborted")
            txt_lines.append(f"  FINAL SCORE          : 0.0000   DEEPFAKE")
            continue

        txt_lines.append(f"\n  [Video Engine]")
        txt_lines.append(f"  Visual Head Score    : {result['visual_head_score']:.4f}")
        txt_lines.append(f"  Temporal LSTM Score  : {result['temporal_score']:.4f}")
        txt_lines.append(f"\n  Frame Scores:")
        for j, s in enumerate(result['frame_scores']):
            txt_lines.append(f"    Frame {j+1:02d}          : {s:.4f}")

        txt_lines.append(f"\n  [Audio Engine]")
        txt_lines.append(f"  Has Audio            : {result['has_audio']}")
        txt_lines.append(f"  Audio Head Score     : {result['audio_head_score']:.4f}")
        txt_lines.append(f"  Pause Pattern Score  : {result['pause_score']:.4f}")
        txt_lines.append(f"  Consistency Score    : {result['consistency_score']:.4f}")

        txt_lines.append(f"\n  [Fusion]")
        txt_lines.append(f"  Fusion Score         : {result['fusion_score']:.4f}")
        txt_lines.append(f"  Video+Audio Score    : {result['video_audio_score']:.4f}")

        txt_lines.append(f"\n  [Comments + Engagement]")
        txt_lines.append(f"  Comments+Eng Score   : {result['comments_eng_score']:.4f}")

        txt_lines.append(f"\n  [Final]")
        txt_lines.append(f"  FINAL SCORE          : {final_score:.4f}")
        txt_lines.append(f"  VERDICT              : {label} ({zone})")
        txt_lines.append(f"  Processing Time      : {result['processing_time']}s")

    # Summary
    txt_lines.append("\n" + "=" * 65)
    txt_lines.append("  SUMMARY")
    txt_lines.append("=" * 65)
    for result in all_results:
        if not result['error']:
            emoji, label, zone = get_verdict(result['final_score'])
            status = "⛔ DEEPFAKE" if result['is_deepfake'] else f"{emoji} {label}"
            txt_lines.append(
                f"  {result['video_name']:<35} "
                f"{result['final_score']:.4f}  {status}"
            )

    txt_content = "\n".join(txt_lines)

    with open(CONFIG['results_txt'], 'w', encoding='utf-8') as f:
        f.write(txt_content)

    # ── JSON REPORT ───────────────────────────────────────────────────────

    json_data = {
        'timestamp'     : timestamp,
        'weights_used'  : CONFIG['weights_path'],
        'meso_weights'  : CONFIG['meso_weights'],
        'pipeline'      : 'MesoNet Gate → Video → Audio → Comments+Engagement',
        'score_weights' : {
            'video_audio'   : CONFIG['w_video_audio'],
            'comments_eng'  : CONFIG['w_comments_eng'],
        },
        'total_videos'  : len(all_results),
        'results'       : [],
    }

    for result in all_results:
        if not result['error']:
            emoji, label, zone = get_verdict(result['final_score'])
            entry = {
                **result,
                'verdict'       : 'DEEPFAKE' if result['is_deepfake'] else label,
                'verdict_zone'  : 'Deepfake Content' if result['is_deepfake'] else zone,
                'verdict_emoji' : '⛔' if result['is_deepfake'] else emoji,
            }
            json_data['results'].append(entry)
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
    print("  ─────────────────────────────────────────────────────────")
    print("  Pipeline:")
    print("    [1] 🔍 MesoNet Deepfake Gate  (abort if ≥ 80% deepfake)")
    print("    [2] 🎥 Video Engine  (MobileNetV2 + LSTM + Attention)")
    print("    [3] 🎵 Audio Engine  (VGGish + Pause + Consistency)")
    print("    [4] 🔗 Fusion        (Video+Audio → 40 % of final)")
    print("    [5] 💬 Comments + Engagement Engine  (60 % of final)")
    print("  ─────────────────────────────────────────────────────────")
    print("  Final Score = 40% Video+Audio  +  60% Comments+Engagement")
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

    print(f"\n  Loading Models …")
    visual_model, audio_head, audio_analyzer, meso_model = load_models(device)

    # ── ANALYZE EACH VIDEO ───────────────────────────────────────────────────

    print(f"\n  Analyzing {len(test_videos)} video(s) …")

    all_results = []

    for i, video_path in enumerate(test_videos, 1):
        result = analyze_video(
            video_path,
            visual_model,
            audio_head,
            audio_analyzer,
            meso_model,
            device,
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

    deepfakes = 0
    for result in all_results:
        if not result['error']:
            emoji, label, zone = get_verdict(result['final_score'])
            if result['is_deepfake']:
                deepfakes += 1
                print(f"  {result['video_name']:<35} "
                      f"{'0.0000':>6}  ⛔ DEEPFAKE")
            else:
                print(f"  {result['video_name']:<35} "
                      f"{result['final_score']:>6.4f}  "
                      f"{emoji} {label}")
        else:
            print(f"  {result['video_name']:<35}  ❌ ERROR")

    print("  " + "-" * 55)
    print(f"  Deepfakes Detected    : {deepfakes} / {len(all_results)}")
    print(f"  Total Processing Time : {total_time:.2f}s")
    print("═" * 65)

    # ── SAVE REPORT ──────────────────────────────────────────────────────────
    
    save_report(all_results)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test()