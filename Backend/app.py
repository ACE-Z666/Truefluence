"""
app.py  —  TrueFluence Flask API Backend
==========================================
Endpoints:
    POST /api/analyze   — upload video + comments + engagement data
    GET  /api/reels     — list all analysed reels
    GET  /              — serve frontend

Usage:
    cd d:\\Truefluence\\Backend
    python app.py
"""

import os
import sys
import json
import uuid
import time
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── path setup so we can import the Multimodals pipeline ──────────────────────
ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTIMODALS    = os.path.join(ROOT_DIR, 'Multimodals')
FRONTEND_DIR   = os.path.join(ROOT_DIR, 'Frontened')
UPLOAD_DIR     = os.path.join(ROOT_DIR, 'Backend', 'uploads')
REELS_DB       = os.path.join(ROOT_DIR, 'Backend', 'reels_db.json')

os.makedirs(UPLOAD_DIR, exist_ok=True)
sys.path.insert(0, MULTIMODALS)
sys.path.insert(0, os.path.join(MULTIMODALS, 'comments'))

import torch
from visual_engine import VisualQualityHead, extract_quality_frames
from audio_engine   import AdvancedAudioAnalyzer
from train          import AudioClassificationHead, extract_visual_features, extract_audio_features
from mesonet        import load_meso4, screen_for_deepfake

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_PATH      = os.path.join(MULTIMODALS, 'models', 'weights', 'best_model.pth')
MESO_WEIGHTS      = os.path.join(MULTIMODALS, 'models', 'weights', 'meso4_DF.pth')
DEEPFAKE_THRESH   = 0.80
W_VIDEO_AUDIO     = 0.40
W_COMMENTS_ENG    = 0.60
AUDIO_DROPOUT     = 0.3

THRESHOLDS = {'scam': 0.3, 'likely_scam': 0.5, 'uncertain': 0.7}

# ─────────────────────────────────────────────────────────────────────────────
# MODELS  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Loading TrueFluence models …")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt         = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
visual_model = VisualQualityHead().to(device)
visual_model.head.load_state_dict(ckpt['head'])
visual_model.temporal_lstm.load_state_dict(ckpt['temporal_lstm'])
visual_model.temporal_attention.load_state_dict(ckpt['temporal_attention'])
visual_model.temporal_classifier.load_state_dict(ckpt['temporal_classifier'])
visual_model.fusion_network.load_state_dict(ckpt['fusion_network'])
visual_model.eval()

audio_head = AudioClassificationHead(dropout=AUDIO_DROPOUT).to(device)
audio_head.load_state_dict(ckpt['audio_head'])
audio_head.eval()

audio_analyzer = AdvancedAudioAnalyzer(device=str(device))
meso_model     = load_meso4(device, MESO_WEIGHTS)

print("  ✅ All models ready\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_verdict(score):
    if score <= THRESHOLDS['scam']:       return '🔴', 'SCAM',        'Confident Fake'
    if score <= THRESHOLDS['likely_scam']:return '🟠', 'LIKELY SCAM', 'Suspicious'
    if score <= THRESHOLDS['uncertain']:  return '🟡', 'UNCERTAIN',   'Borderline'
    return '🟢', 'REAL', 'Confident Real'

def load_reels_db():
    if os.path.exists(REELS_DB):
        with open(REELS_DB, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_reels_db(reels):
    with open(REELS_DB, 'w', encoding='utf-8') as f:
        json.dump(reels, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path, comments_list, followers, likes, num_comments):
    """
    Full TrueFluence pipeline.

    Returns dict with all scores + final verdict.
    """
    result = {
        'is_deepfake'        : False,
        'deepfake_prob'      : 0.0,
        'visual_head_score'  : 0.0,
        'temporal_score'     : 0.0,
        'audio_head_score'   : 0.0,
        'fusion_score'       : 0.0,
        'video_audio_score'  : 0.0,
        'comments_eng_score' : 0.5,
        'final_score'        : 0.0,
        'has_audio'          : False,
        'frame_scores'       : [],
        'error'              : None,
    }

    try:
        # ── MesoNet deepfake gate ──────────────────────────────────────────
        meso_r = screen_for_deepfake(video_path, meso_model, device,
                                     num_frames=16, threshold=DEEPFAKE_THRESH)
        result['is_deepfake']  = meso_r['is_deepfake']
        result['deepfake_prob']= meso_r['deepfake_prob']

        if meso_r['is_deepfake']:
            result['final_score'] = 0.0
            return result

        with torch.no_grad():
            # ── Video engine ───────────────────────────────────────────────
            frames, frames_batch, frame_vecs = extract_visual_features(
                video_path, visual_model, device)

            raw_logits   = visual_model.head[:-1](frame_vecs)
            frame_scores = torch.sigmoid(raw_logits).squeeze(1)
            avg_logit         = raw_logits.mean().unsqueeze(0)
            visual_head_score = torch.sigmoid(avg_logit).item()

            result['frame_scores']      = [round(s, 4) for s in frame_scores.cpu().numpy().tolist()]
            result['visual_head_score'] = round(visual_head_score, 4)

            temporal_score = visual_model.forward_temporal(frames_batch).item()
            result['temporal_score'] = round(temporal_score, 4)

            # ── Audio engine ───────────────────────────────────────────────
            vggish_emb, audio_vector, has_audio = extract_audio_features(
                video_path, audio_analyzer, device)
            result['has_audio'] = has_audio

            if has_audio:
                audio_head_score = torch.sigmoid(audio_head(vggish_emb)).item()
                result['audio_head_score'] = round(audio_head_score, 4)
            else:
                audio_head_score = 0.0

            # ── Fusion ─────────────────────────────────────────────────────
            visual_vec            = frame_vecs.mean(dim=0)[:135].unsqueeze(0)
            audio_padded          = torch.zeros(1, 135, device=device)
            audio_padded[0, :130] = audio_vector
            fusion_score = visual_model.forward_fusion(visual_vec, audio_padded).item()
            result['fusion_score'] = round(fusion_score, 4)

            if has_audio:
                video_audio_score = (0.15 * visual_head_score + 0.15 * temporal_score
                                   + 0.15 * audio_head_score  + 0.55 * fusion_score)
            else:
                video_audio_score = 0.40 * visual_head_score + 0.60 * temporal_score
            result['video_audio_score'] = round(video_audio_score, 4)

        # ── Comments + Engagement engine ───────────────────────────────────
        try:
            from bert_comment.run_bert_comments import analyze_comments_bert
            from bert_comment.run_engagement    import analyze_engagement

            if comments_list:
                cr = analyze_comments_bert(comments_list)
                comment_score = cr['comment_authenticity_score']
            else:
                comment_score = 0.5

            eng_score          = analyze_engagement(followers, likes, num_comments)
            comments_eng_score = round(0.5 * comment_score + 0.5 * eng_score, 4)
            result['comments_eng_score'] = comments_eng_score

        except Exception as ce:
            print(f"  Comments engine error: {ce}")
            result['comments_eng_score'] = 0.5

        # ── Final score ─────────────────────────────────────────────────────
        final_score = (W_VIDEO_AUDIO  * result['video_audio_score'] +
                       W_COMMENTS_ENG * result['comments_eng_score'])
        result['final_score'] = round(final_score, 4)

    except Exception as e:
        result['error'] = str(e)
        print(f"  Pipeline error: {e}")

    return result

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/api/reels', methods=['GET'])
def get_reels():
    return jsonify(load_reels_db())

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Accepts multipart/form-data:
        video       : file
        username    : str
        caption     : str
        followers   : int
        likes       : int
        num_comments: int
        comments    : JSON array of strings  (up to 10)
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # ── Save video ────────────────────────────────────────────────────────
    reel_id    = str(uuid.uuid4())[:8]
    ext        = os.path.splitext(video_file.filename)[1].lower() or '.mp4'
    saved_name = f"{reel_id}{ext}"
    video_path = os.path.join(UPLOAD_DIR, saved_name)
    video_file.save(video_path)

    # ── Parse metadata ────────────────────────────────────────────────────
    username     = request.form.get('username', 'anonymous')
    caption      = request.form.get('caption',  'No caption')
    followers    = int(request.form.get('followers',    0))
    likes        = int(request.form.get('likes',        0))
    num_comments = int(request.form.get('num_comments', 0))

    raw_comments = request.form.get('comments', '[]')
    try:
        comments_list = json.loads(raw_comments)
    except Exception:
        comments_list = []

    # ── Run pipeline ──────────────────────────────────────────────────────
    print(f"\n  Analyzing reel {reel_id} ({video_file.filename}) …")
    t0      = time.time()
    pipeline = run_pipeline(video_path, comments_list, followers, likes, num_comments)
    elapsed = round(time.time() - t0, 2)

    emoji, label, zone = get_verdict(pipeline['final_score'])
    if pipeline['is_deepfake']:
        emoji, label, zone = '⛔', 'DEEPFAKE', 'Deepfake Detected'

    # ── Build reel record ─────────────────────────────────────────────────
    reel = {
        'id'              : reel_id,
        'username'        : username,
        'caption'         : caption,
        'video_url'       : f'/uploads/{saved_name}',
        'followers'       : followers,
        'likes'           : likes,
        'num_comments'    : num_comments,
        'comments'        : comments_list,
        'timestamp'       : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processing_time' : elapsed,
        'analysis'        : {
            **pipeline,
            'verdict'       : label,
            'verdict_emoji' : emoji,
            'verdict_zone'  : zone,
        }
    }

    # ── Persist ───────────────────────────────────────────────────────────
    reels = load_reels_db()
    reels.insert(0, reel)          # newest first
    save_reels_db(reels)

    print(f"  ✅ Done in {elapsed}s — {emoji} {label}")
    return jsonify(reel)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("  🚀 TrueFluence API starting …")
    print(f"  Frontend : {FRONTEND_DIR}")
    print(f"  Uploads  : {UPLOAD_DIR}")
    app.run(host='0.0.0.0', port=5000, debug=False)
