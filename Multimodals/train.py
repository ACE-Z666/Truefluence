import os
import time
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from visual_engine import VisualQualityHead, extract_quality_frames
from audio_engine  import AdvancedAudioAnalyzer, AudioFeatureExtractor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    'real_dir'          : os.path.join('dataset', 'real_videos'),
    'scam_dir'          : os.path.join('dataset', 'scam_videos'),
    'weights_dir'       : os.path.join('models', 'weights'),

    # Per-phase best model paths
    'best_visual_head'      : os.path.join('models', 'weights', 'best_visual_head.pth'),
    'best_visual_temporal'  : os.path.join('models', 'weights', 'best_visual_temporal.pth'),
    'best_audio_head'       : os.path.join('models', 'weights', 'best_audio_head.pth'),
    'best_fusion'           : os.path.join('models', 'weights', 'best_fusion.pth'),

    # Overall best + final
    'best_model'        : os.path.join('models', 'weights', 'best_model.pth'),
    'final_model'       : os.path.join('models', 'weights', 'final_model.pth'),

    # ── UPDATED FOR SMALL DATASET (15 videos) ────────────────────────────
    'max_epochs_per_phase'  : 5,        # ← was 30, small dataset = few epochs
    'early_stop_patience'   : 5,        # ← was 3,  give more chances on tiny val set
    'num_frames'            : 8,        # Frames extracted per video
    'val_split'             : 0.2,      # 80/20 stratified split
    'random_seed'           : 42,

    # Learning rates
    'lr_visual_head'        : 1e-3,
    'lr_visual_temporal'    : 5e-4,
    'lr_audio_head'         : 1e-3,
    'lr_fusion'             : 5e-4,

    # Dropout (increased to fight overfitting on small dataset)
    'audio_dropout'         : 0.5,      # ← was 0.3, higher = less overfit

    # Video formats
    'video_formats'         : ['*.mp4', '*.avi', '*.mov', '*.mkv'],

    # No-audio fallback
    'no_audio_score'        : 0.0,      # If video has no audio → score = 0.0
}

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO CLASSIFICATION HEAD
# Defined here in train.py (self-contained, no changes to audio_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

class AudioClassificationHead(nn.Module):
    """
    Simple trainable head on top of frozen VGGish embeddings.

    Architecture:
        VGGish (128-dim, FROZEN) → this head → fake/real prediction

        Linear(128 → 64) → ReLU → Dropout(0.3)
        Linear(64  →  1) → Raw Logit

    Input : 128-dim VGGish embedding
    Output: Raw logit (apply sigmoid for probability)
            logit > 0 → real audio
            logit < 0 → fake audio
    """
    def __init__(self, dropout=0.3):
        super(AudioClassificationHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)            # Raw logit output
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, vggish_embedding):
        """
        Args:
            vggish_embedding: (batch, 128) or (128,) tensor
        Returns:
            logit: (batch, 1) or (1,) raw logit
        """
        if vggish_embedding.dim() == 1:
            vggish_embedding = vggish_embedding.unsqueeze(0)
        return self.head(vggish_embedding)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def collect_videos():
    """
    Scan real_videos/ and scam_videos/ directories.
    Returns:
        videos : list of video file paths
        labels : list of float labels (1.0=real, 0.0=scam)
    """
    videos, labels = [], []

    real_videos, scam_videos = [], []

    if os.path.exists(CONFIG['real_dir']):
        for fmt in CONFIG['video_formats']:
            real_videos += glob.glob(os.path.join(CONFIG['real_dir'], fmt))
    else:
        print(f"  ⚠  Real dir not found : {CONFIG['real_dir']}")

    if os.path.exists(CONFIG['scam_dir']):
        for fmt in CONFIG['video_formats']:
            scam_videos += glob.glob(os.path.join(CONFIG['scam_dir'], fmt))
    else:
        print(f"  ⚠  Scam dir not found : {CONFIG['scam_dir']}")

    videos = real_videos + scam_videos
    labels = [1.0] * len(real_videos) + [0.0] * len(scam_videos)

    print(f"  Real  videos : {len(real_videos)}")
    print(f"  Scam  videos : {len(scam_videos)}")
    print(f"  Total        : {len(videos)}")

    return videos, labels


def stratified_split(videos, labels):
    """
    Stratified 80/20 split.
    Guarantees equal real/scam ratio in both train and val sets.
    """
    videos_arr = np.array(videos)
    labels_arr = np.array(labels)

    splitter = StratifiedShuffleSplit(
        n_splits     = 1,
        test_size    = CONFIG['val_split'],
        random_state = CONFIG['random_seed']
    )

    for train_idx, val_idx in splitter.split(videos_arr, labels_arr.astype(int)):
        train_videos = videos_arr[train_idx].tolist()
        train_labels = labels_arr[train_idx].tolist()
        val_videos   = videos_arr[val_idx].tolist()
        val_labels   = labels_arr[val_idx].tolist()

    print(f"\n  Stratified 80/20 Split:")
    print(f"  Train : {len(train_videos)} "
          f"(real={int(sum(train_labels))}, "
          f"scam={int(len(train_labels) - sum(train_labels))})")
    print(f"  Val   : {len(val_videos)} "
          f"(real={int(sum(val_labels))}, "
          f"scam={int(len(val_labels) - sum(val_labels))})")

    return train_videos, train_labels, val_videos, val_labels


def compute_pos_weight(labels, device):
    """
    pos_weight = num_real / num_scam
    Penalizes missing a scam more than a false alarm.
    Automatically adapts to dataset imbalance.
    """
    num_real = sum(1 for l in labels if l == 1.0)
    num_scam = sum(1 for l in labels if l == 0.0)

    if num_scam == 0:
        print("  ⚠  No scam videos → pos_weight = 1.0")
        return torch.tensor([1.0], device=device)

    pw = num_real / num_scam
    print(f"  pos_weight = {num_real}/{num_scam} = {pw:.3f}")
    print(f"  (Missing a scam costs {pw:.1f}x more than a false alarm)")
    return torch.tensor([pw], dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_visual_features(video_path, model, device):
    """
    Extract frames and run through frozen MobileNetV2 backbone.

    Returns:
        frames        : (N, 3, 224, 224)   raw frames
        frames_batch  : (1, N, 3, 224, 224) batched for temporal
        frame_vecs    : (N, 1280)           backbone feature vectors
    """
    frames = extract_quality_frames(
        video_path,
        num_frames = CONFIG['num_frames']
    ).to(device)

    # frames shape: (N, 3, 224, 224)
    if frames.dim() == 3:
        # Missing channel dim → (1, 3, H, W) edge case
        frames = frames.unsqueeze(0)

    # Ensure exactly 4D: (N, 3, H, W)
    if frames.dim() != 4:
        raise ValueError(
            f"Unexpected frames shape: {frames.shape}, "
            f"expected (N, 3, H, W)"
        )

    # Backbone forward (frozen)
    with torch.no_grad():
        feats  = model.backbone(frames)                             # (N, 1280, 1, 1)
        pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1))  # (N, 1280, 1, 1)
        vecs   = torch.flatten(pooled, 1)                          # (N, 1280)

    # Add batch dim for temporal: (N, 3, 224, 224) → (1, N, 3, 224, 224)
    frames_batch = frames.unsqueeze(0)                             # (1, N, 3, 224, 224)

    return frames, frames_batch, vecs


def extract_audio_features(video_path, audio_analyzer, device):
    """
    Extract audio embedding + pause score from video audio.

    Uses moviepy pipeline via audio_analyzer.feature_extractor
    for ALL audio operations — no direct librosa.load(.mp4)

    Returns:
        vggish_embedding : (128,) tensor   → AudioClassificationHead
        audio_vector     : (130,) tensor   → Fusion network
        has_audio        : bool
    """
    try:
        # ── Step 1: Extract 128-dim features via moviepy pipeline ────
        # process_audio() internally does:
        #   moviepy → wav → librosa → 128-dim features
        vggish_np = audio_analyzer.feature_extractor.process_audio(video_path)

        if vggish_np is None or len(vggish_np) == 0:
            raise ValueError("No audio or empty feature output")

        # ── Step 2: Pause + authenticity score via moviepy pipeline ──
        # analyze_voice_authenticity() also uses moviepy internally
        voice_result = audio_analyzer.analyze_voice_authenticity(video_path)
        pause_score  = voice_result.get(
            'authenticity_score',
            CONFIG['no_audio_score']
        )

        # ── Step 3: Temporal consistency via moviepy pipeline ─────────
        # FIX: was using librosa.load(video_path) directly → FAILS
        # NOW: extract wav first via moviepy, then load wav with librosa
        wav_path = audio_analyzer.feature_extractor._extract_wav_from_video(
            video_path
        )

        if wav_path is not None:
            try:
                y, sr     = librosa.load(
                    wav_path,               # ← .wav not .mp4 ✓
                    sr       = 16000,
                    mono     = True,
                    duration = 30
                )
                temporal          = audio_analyzer.feature_extractor\
                                        .extract_temporal_features(y, sr)
                consistency_score = temporal.get(
                    'temporal_consistency',
                    CONFIG['no_audio_score']
                )
            except Exception:
                consistency_score = CONFIG['no_audio_score']
            finally:
                # Always clean up temp wav
                import os as _os
                if _os.path.exists(wav_path):
                    try:
                        _os.unlink(wav_path)
                    except Exception:
                        pass
        else:
            consistency_score = CONFIG['no_audio_score']

        # ── Step 4: Build tensors ──────────────────────────────────────
        # VGGish embedding tensor (128,)
        vggish_embedding = torch.tensor(
            vggish_np,
            dtype = torch.float32
        ).to(device)

        # Full audio vector (130,)
        # [0:128] features | [128] pause_score | [129] consistency_score
        audio_vector = torch.tensor(
            np.concatenate([
                vggish_np,
                [pause_score],
                [consistency_score]
            ]).astype(np.float32)
        ).to(device)

        return vggish_embedding, audio_vector, True

    except Exception as e:
        # No audio or extraction failed → fallback zeros
        vggish_embedding = torch.zeros(128, device=device)
        audio_vector     = torch.zeros(130, device=device)
        return vggish_embedding, audio_vector, False


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops a training phase when val_loss stops improving.
    patience = 3 → stops if no improvement for 3 consecutive epochs.
    """
    def __init__(self, patience=3, phase_name=''):
        self.patience    = patience
        self.phase_name  = phase_name
        self.best_loss   = float('inf')
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0                          # Reset counter on improvement
            return True                                 # Improved → save model
        else:
            self.counter += 1
            print(f"  ⏳ No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  🛑 Early stopping triggered for {self.phase_name}")
            return False                                # No improvement


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, val_loss, val_acc, **state_dicts):
    """
    Save model state dicts with metadata.
    Converts all values to Python natives to avoid
    PyTorch 2.6 weights_only unpickling error.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch'    : int(epoch),        # ← Python int, not numpy
        'val_loss' : float(val_loss),   # ← Python float, not numpy
        'val_acc'  : float(val_acc),    # ← Python float, not numpy
        **state_dicts
    }, path)


def freeze_module(module):
    """Freeze all parameters in a module (no gradient updates)."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: VISUAL QUALITY HEAD TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def phase1_visual_head(
    model, train_videos, train_labels,
    val_videos, val_labels,
    criterion, device
):
    """
    Phase 1: Train Visual Quality Head ONLY.

    What trains : model.head (Linear layers)
    What freezes: model.backbone (MobileNetV2)
                  model.temporal_lstm
                  model.temporal_attention
                  model.fusion_network

    Input  : (num_frames, 3, 224, 224) frames
    Forward: backbone(frozen) → head(trainable) → logit
    Loss   : BCEWithLogitsLoss(pos_weight)
    """
    print("\n" + "═" * 70)
    print("  PHASE 1 — Visual Quality Head")
    print("  Trains  : model.head")
    print("  Freezes : backbone, temporal, fusion")
    print("═" * 70)

    # Freeze everything except head
    freeze_module(model.backbone)
    freeze_module(model.temporal_lstm)
    freeze_module(model.temporal_attention)
    freeze_module(model.temporal_classifier)
    freeze_module(model.fusion_network)
    unfreeze_module(model.head)

    optimizer    = optim.Adam(model.head.parameters(), lr=CONFIG['lr_visual_head'])
    early_stop   = EarlyStopping(patience=CONFIG['early_stop_patience'], phase_name='Phase 1')
    best_val_loss = float('inf')

    for epoch in range(CONFIG['max_epochs_per_phase']):
        model.train()

        # Shuffle
        combined = list(zip(train_videos, train_labels))
        random.shuffle(combined)
        ep_videos, ep_labels = zip(*combined)

        train_losses, correct, total = [], 0, 0

        for video_path, label in tqdm(
            zip(ep_videos, ep_labels),
            total  = len(ep_videos),
            desc   = f"  P1 Epoch {epoch+1}",
            ncols  = 70
        ):
            try:
                target = torch.tensor([label], dtype=torch.float32).to(device)

                # Extract visual features (backbone frozen)
                _, _, frame_vecs = extract_visual_features(video_path, model, device)
                # frame_vecs: (num_frames, 1280)

                optimizer.zero_grad()

                # Head forward: (num_frames, 1280) → (num_frames, 1)
                raw_logits = model.head[:-1](frame_vecs)   # All layers except Sigmoid
                avg_logit  = raw_logits.mean().unsqueeze(0) # Scalar → (1,)

                loss = criterion(avg_logit, target)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                # Accuracy
                pred      = 1.0 if torch.sigmoid(avg_logit).item() > 0.5 else 0.0
                correct  += int(pred == label)
                total    += 1

            except Exception as e:
                tqdm.write(f"  ⚠  {os.path.basename(video_path)}: {e}")
                continue

        # Validation
        val_loss, val_acc = _validate_visual_head(
            model, val_videos, val_labels, criterion, device
        )

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        train_acc      = correct / total        if total > 0   else 0.0

        print(f"\n  P1 Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Early stopping check
        improved = early_stop.step(val_loss)
        if improved:
            best_val_loss = val_loss
            save_checkpoint(
                CONFIG['best_visual_head'],
                epoch + 1, val_loss, val_acc,
                head = model.head.state_dict()
            )
            print(f"  ✅ Phase 1 best model saved (val_loss={val_loss:.4f})")

        if early_stop.should_stop:
            break

    # ── FIX: weights_only=True ───────────────────────────────────────────
    ckpt = torch.load(
        CONFIG['best_visual_head'],
        map_location = device,
        weights_only = True             # ← FIX UnpicklingError
    )
    model.head.load_state_dict(ckpt['head'])
    print(f"\n  Phase 1 complete. Best val_loss: {early_stop.best_loss:.4f}")
    print(f"  Freezing model.head for Phase 2...")
    freeze_module(model.head)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: VISUAL TEMPORAL (LSTM + ATTENTION) TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def phase2_visual_temporal(
    model, train_videos, train_labels,
    val_videos, val_labels,
    criterion, device
):
    """
    Phase 2: Train Visual Temporal LSTM + Attention ONLY.

    What trains : model.temporal_lstm
                  model.temporal_attention
                  model.temporal_classifier
    What freezes: model.backbone (MobileNetV2)   [from Phase 1]
                  model.head                     [frozen in Phase 1]
                  model.fusion_network

    Input  : enriched frame features (1294-dim per frame)
    Forward: LSTM → Attention → classifier → logit
    Loss   : BCEWithLogitsLoss(pos_weight)
    """
    print("\n" + "═" * 70)
    print("  PHASE 2 — Visual Temporal (LSTM + Attention)")
    print("  Trains  : temporal_lstm, temporal_attention, temporal_classifier")
    print("  Freezes : backbone, head (Phase 1 frozen), fusion")
    print("═" * 70)

    # Unfreeze temporal only
    unfreeze_module(model.temporal_lstm)
    unfreeze_module(model.temporal_attention)
    unfreeze_module(model.temporal_classifier)
    freeze_module(model.fusion_network)

    temporal_params = (
        list(model.temporal_lstm.parameters())       +
        list(model.temporal_attention.parameters())  +
        list(model.temporal_classifier.parameters())
    )

    optimizer  = optim.Adam(temporal_params, lr=CONFIG['lr_visual_temporal'])
    early_stop = EarlyStopping(patience=CONFIG['early_stop_patience'], phase_name='Phase 2')

    for epoch in range(CONFIG['max_epochs_per_phase']):
        model.train()

        combined = list(zip(train_videos, train_labels))
        random.shuffle(combined)
        ep_videos, ep_labels = zip(*combined)

        train_losses, correct, total = [], 0, 0

        for video_path, label in tqdm(
            zip(ep_videos, ep_labels),
            total  = len(ep_videos),
            desc   = f"  P2 Epoch {epoch+1}",
            ncols  = 70
        ):
            try:
                target       = torch.tensor([label], dtype=torch.float32).to(device)
                frames, frames_batch, _ = extract_visual_features(
                    video_path, model, device
                )

                optimizer.zero_grad()

                # forward_temporal → sigmoid output (1, 1)
                temporal_score   = model.forward_temporal(frames_batch)
                clamped          = torch.clamp(temporal_score, 1e-6, 1 - 1e-6)
                temporal_logit   = torch.log(clamped / (1 - clamped))

                loss = criterion(temporal_logit, target.unsqueeze(0))
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                pred      = 1.0 if temporal_score.item() > 0.5 else 0.0
                correct  += int(pred == label)
                total    += 1

            except Exception as e:
                tqdm.write(f"  ⚠  {os.path.basename(video_path)}: {e}")
                continue

        val_loss, val_acc = _validate_temporal(
            model, val_videos, val_labels, criterion, device
        )

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        train_acc      = correct / total        if total > 0   else 0.0

        print(f"\n  P2 Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        improved = early_stop.step(val_loss)
        if improved:
            save_checkpoint(
                CONFIG['best_visual_temporal'],
                epoch + 1, val_loss, val_acc,
                temporal_lstm         = model.temporal_lstm.state_dict(),
                temporal_attention    = model.temporal_attention.state_dict(),
                temporal_classifier   = model.temporal_classifier.state_dict()
            )
            print(f"  ✅ Phase 2 best model saved (val_loss={val_loss:.4f})")

        if early_stop.should_stop:
            break

    ckpt = torch.load(
        CONFIG['best_visual_temporal'],
        map_location = device,
        weights_only = True             # ← FIX
    )
    model.temporal_lstm.load_state_dict(ckpt['temporal_lstm'])
    model.temporal_attention.load_state_dict(ckpt['temporal_attention'])
    model.temporal_classifier.load_state_dict(ckpt['temporal_classifier'])
    print(f"\n  Phase 2 complete. Best val_loss: {early_stop.best_loss:.4f}")
    print(f"  Freezing temporal components for Phase 3...")
    freeze_module(model.temporal_lstm)
    freeze_module(model.temporal_attention)
    freeze_module(model.temporal_classifier)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: AUDIO HEAD TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def phase3_audio_head(
    audio_head, audio_analyzer,
    train_videos, train_labels,
    val_videos, val_labels,
    criterion, device
):
    """
    Phase 3: Train Audio Classification Head ONLY.

    What trains : audio_head (Linear 128→64→1)
    What freezes: VGGish backbone (always frozen in audio_engine.py)
                  ALL visual model components (frozen from Phases 1 & 2)

    Input  : VGGish embedding (128-dim, frozen VGGish output)
    Forward: audio_head → logit
    Loss   : BCEWithLogitsLoss(pos_weight)

    No audio → score = 0.0 (CONFIG['no_audio_score'])
    Label   : same as video label (real=1.0, scam=0.0)
    """
    print("\n" + "═" * 70)
    print("  PHASE 3 — Audio Classification Head")
    print("  Trains  : AudioClassificationHead (128→64→1)")
    print("  Freezes : VGGish (always frozen), all visual components")
    print("  Fallback: No audio → score = 0.0")
    print("═" * 70)

    optimizer  = optim.Adam(audio_head.parameters(), lr=CONFIG['lr_audio_head'])
    early_stop = EarlyStopping(patience=CONFIG['early_stop_patience'], phase_name='Phase 3')

    for epoch in range(CONFIG['max_epochs_per_phase']):
        audio_head.train()

        combined = list(zip(train_videos, train_labels))
        random.shuffle(combined)
        ep_videos, ep_labels = zip(*combined)

        train_losses    = []
        correct, total  = 0, 0
        skipped_audio   = 0

        for video_path, label in tqdm(
            zip(ep_videos, ep_labels),
            total  = len(ep_videos),
            desc   = f"  P3 Epoch {epoch+1}",
            ncols  = 70
        ):
            try:
                target = torch.tensor([label], dtype=torch.float32).to(device)

                # Extract VGGish embedding (VGGish frozen inside audio_engine)
                vggish_emb, _, has_audio = extract_audio_features(
                    video_path, audio_analyzer, device
                )

                if not has_audio:
                    # No audio → score = 0.0, skip gradient update
                    skipped_audio += 1
                    pred      = 1.0 if CONFIG['no_audio_score'] > 0.5 else 0.0
                    correct  += int(pred == label)
                    total    += 1
                    continue

                optimizer.zero_grad()

                # Audio head forward
                logit = audio_head(vggish_emb)          # (1, 1)
                loss  = criterion(logit, target.unsqueeze(0))
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                pred      = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
                correct  += int(pred == label)
                total    += 1

            except Exception as e:
                tqdm.write(f"  ⚠  {os.path.basename(video_path)}: {e}")
                continue

        val_loss, val_acc = _validate_audio_head(
            audio_head, audio_analyzer,
            val_videos, val_labels, criterion, device
        )

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        train_acc      = correct / total        if total > 0   else 0.0

        print(f"\n  P3 Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"No-Audio: {skipped_audio}")

        improved = early_stop.step(val_loss)
        if improved:
            save_checkpoint(
                CONFIG['best_audio_head'],
                epoch + 1, val_loss, val_acc,
                audio_head = audio_head.state_dict()
            )
            print(f"  ✅ Phase 3 best model saved (val_loss={val_loss:.4f})")

        if early_stop.should_stop:
            break

    # ── Guard: only load if checkpoint was actually saved ────────────
    if os.path.exists(CONFIG['best_audio_head']):
        ckpt = torch.load(
            CONFIG['best_audio_head'],
            map_location = device,
            weights_only = True         # ← FIX 1: was missing weights_only
        )
        audio_head.load_state_dict(ckpt['audio_head'])
        print(f"\n  Phase 3 best weights restored ✓")
    else:
        print(f"\n  ⚠  No audio checkpoint saved")
        print(f"     All videos had no audio → skipping audio head load")
        print(f"     Continuing to Phase 4 with untrained audio head")

    print(f"\n  Phase 3 complete. Best val_loss: {early_stop.best_loss:.4f}")
    print(f"  Freezing audio_head for Phase 4...")
    freeze_module(audio_head)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: FUSION NETWORK TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def phase4_fusion(
    model, audio_head, audio_analyzer,
    train_videos, train_labels,
    val_videos, val_labels,
    criterion, device
):
    """
    Phase 4: Train Fusion Network ONLY.

    What trains : model.fusion_network
    What freezes: model.backbone          [Phase 1]
                  model.head              [Phase 1]
                  model.temporal_lstm     [Phase 2]
                  model.temporal_attention[Phase 2]
                  model.temporal_classifier[Phase 2]
                  audio_head              [Phase 3]
                  VGGish                  [always]

    Input  : visual_features (135-dim) + audio_vector (130-dim)
    Forward: fusion_network → final score
    Loss   : BCEWithLogitsLoss(pos_weight)

    No audio → audio_vector = zeros(130) [score contribution = 0.0]
    """
    print("\n" + "═" * 70)
    print("  PHASE 4 — Fusion Network (Visual + Audio)")
    print("  Trains  : model.fusion_network")
    print("  Freezes : ALL previous components (backbone, head,")
    print("            temporal, audio_head, VGGish)")
    print("  No Audio: audio_vector = zeros → score = 0.0")
    print("═" * 70)

    unfreeze_module(model.fusion_network)

    optimizer  = optim.Adam(
        model.fusion_network.parameters(),
        lr = CONFIG['lr_fusion']
    )
    early_stop     = EarlyStopping(patience=CONFIG['early_stop_patience'], phase_name='Phase 4')
    best_val_loss  = float('inf')
    overall_best   = float('inf')

    for epoch in range(CONFIG['max_epochs_per_phase']):
        model.train()
        audio_head.eval()                       # Audio head frozen, set to eval

        combined = list(zip(train_videos, train_labels))
        random.shuffle(combined)
        ep_videos, ep_labels = zip(*combined)

        train_losses    = []
        correct, total  = 0, 0
        no_audio_count  = 0

        for video_path, label in tqdm(
            zip(ep_videos, ep_labels),
            total  = len(ep_videos),
            desc   = f"  P4 Epoch {epoch+1}",
            ncols  = 70
        ):
            try:
                target = torch.tensor([label], dtype=torch.float32).to(device)

                # Visual features
                frames, frames_batch, frame_vecs = extract_visual_features(
                    video_path, model, device
                )

                # Visual representation: mean pool frame vectors → (1, 1280)
                # Reduce to 135-dim for fusion input
                visual_mean = frame_vecs.mean(dim=0)        # (1280,)
                visual_vec  = visual_mean[:135].unsqueeze(0) # (1, 135)

                # Audio features
                vggish_emb, audio_vector, has_audio = extract_audio_features(
                    video_path, audio_analyzer, device
                )

                if not has_audio:
                    no_audio_count += 1
                    # audio_vector already zeros(130) from extract_audio_features

                # Pad audio_vector 130 → 135 for fusion input
                audio_padded        = torch.zeros(1, 135, device=device)
                audio_padded[0,:130] = audio_vector          # (1, 135)

                optimizer.zero_grad()

                # Fusion forward
                fusion_score   = model.forward_fusion(visual_vec, audio_padded) # (1,1)
                clamped        = torch.clamp(fusion_score, 1e-6, 1 - 1e-6)
                fusion_logit   = torch.log(clamped / (1 - clamped))

                loss = criterion(fusion_logit, target.unsqueeze(0))
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                pred      = 1.0 if fusion_score.item() > 0.5 else 0.0
                correct  += int(pred == label)
                total    += 1

            except Exception as e:
                tqdm.write(f"  ⚠  {os.path.basename(video_path)}: {e}")
                continue

        val_loss, val_acc = _validate_fusion(
            model, audio_head, audio_analyzer,
            val_videos, val_labels, criterion, device
        )

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        train_acc      = correct / total        if total > 0   else 0.0

        print(f"\n  P4 Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"No-Audio: {no_audio_count}")

        improved = early_stop.step(val_loss)
        if improved:
            best_val_loss = val_loss
            # Save phase 4 best
            save_checkpoint(
                CONFIG['best_fusion'],
                epoch + 1, val_loss, val_acc,
                fusion_network = model.fusion_network.state_dict()
            )
            # Save overall best model (all components)
            save_checkpoint(
                CONFIG['best_model'],
                epoch + 1, val_loss, val_acc,
                head                  = model.head.state_dict(),
                temporal_lstm         = model.temporal_lstm.state_dict(),
                temporal_attention    = model.temporal_attention.state_dict(),
                temporal_classifier   = model.temporal_classifier.state_dict(),
                fusion_network        = model.fusion_network.state_dict(),
                audio_head            = audio_head.state_dict(),
            )
            print(f"  ✅ Phase 4 best + Overall best model saved "
                  f"(val_loss={val_loss:.4f})")

        if early_stop.should_stop:
            break

    print(f"\n  Phase 4 complete. Best val_loss: {early_stop.best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS (Per Phase)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_visual_head(model, val_videos, val_labels, criterion, device):
    model.eval()
    losses, correct, total = [], 0, 0

    with torch.no_grad():
        for video_path, label in zip(val_videos, val_labels):
            try:
                target              = torch.tensor([label], dtype=torch.float32).to(device)
                _, _, frame_vecs    = extract_visual_features(video_path, model, device)
                raw_logits          = model.head[:-1](frame_vecs)
                avg_logit           = raw_logits.mean().unsqueeze(0)
                loss                = criterion(avg_logit, target)
                losses.append(loss.item())
                pred     = 1.0 if torch.sigmoid(avg_logit).item() > 0.5 else 0.0
                correct += int(pred == label)
                total   += 1
            except Exception:
                continue

    model.train()
    return (
        np.mean(losses) if losses else float('inf'),
        correct / total if total > 0 else 0.0
    )


def _validate_temporal(model, val_videos, val_labels, criterion, device):
    model.eval()
    losses, correct, total = [], 0, 0

    with torch.no_grad():
        for video_path, label in zip(val_videos, val_labels):
            try:
                target              = torch.tensor([label], dtype=torch.float32).to(device)
                _, frames_batch, _  = extract_visual_features(video_path, model, device)
                temporal_score      = model.forward_temporal(frames_batch)
                clamped             = torch.clamp(temporal_score, 1e-6, 1 - 1e-6)
                temporal_logit      = torch.log(clamped / (1 - clamped))
                loss                = criterion(temporal_logit, target.unsqueeze(0))
                losses.append(loss.item())
                pred     = 1.0 if temporal_score.item() > 0.5 else 0.0
                correct += int(pred == label)
                total   += 1
            except Exception:
                continue

    model.train()
    return (
        np.mean(losses) if losses else float('inf'),
        correct / total if total > 0 else 0.0
    )


def _validate_audio_head(
    audio_head, audio_analyzer,
    val_videos, val_labels, criterion, device
):
    audio_head.eval()
    losses, correct, total = [], 0, 0

    with torch.no_grad():
        for video_path, label in zip(val_videos, val_labels):
            try:
                target              = torch.tensor([label], dtype=torch.float32).to(device)
                vggish_emb, _, has_audio = extract_audio_features(
                    video_path, audio_analyzer, device
                )

                if not has_audio:
                    pred     = 1.0 if CONFIG['no_audio_score'] > 0.5 else 0.0
                    correct += int(pred == label)
                    total   += 1
                    continue

                logit = audio_head(vggish_emb)
                loss  = criterion(logit, target.unsqueeze(0))
                losses.append(loss.item())
                pred     = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
                correct += int(pred == label)
                total   += 1
            except Exception:
                continue

    audio_head.train()
    return (
        np.mean(losses) if losses else float('inf'),
        correct / total if total > 0 else 0.0
    )


def _validate_fusion(
    model, audio_head, audio_analyzer,
    val_videos, val_labels, criterion, device
):
    model.eval()
    audio_head.eval()
    losses, correct, total = [], 0, 0

    with torch.no_grad():
        for video_path, label in zip(val_videos, val_labels):
            try:
                target                          = torch.tensor([label], dtype=torch.float32).to(device)
                _, _, frame_vecs                = extract_visual_features(video_path, model, device)
                visual_vec                      = frame_vecs.mean(dim=0)[:135].unsqueeze(0)
                _, audio_vector, _              = extract_audio_features(video_path, audio_analyzer, device)
                audio_padded                    = torch.zeros(1, 135, device=device)
                audio_padded[0, :130]           = audio_vector
                fusion_score                    = model.forward_fusion(visual_vec, audio_padded)
                clamped                         = torch.clamp(fusion_score, 1e-6, 1 - 1e-6)
                fusion_logit                    = torch.log(clamped / (1 - clamped))
                loss                            = criterion(fusion_logit, target.unsqueeze(0))
                losses.append(loss.item())
                pred     = 1.0 if fusion_score.item() > 0.5 else 0.0
                correct += int(pred == label)
                total   += 1
            except Exception:
                continue

    model.train()
    return (
        np.mean(losses) if losses else float('inf'),
        correct / total if total > 0 else 0.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("\n" + "═" * 70)
    print("  🚀 TRUEFLUENCE — MULTIMODAL SCAM DETECTION TRAINING")
    print("     Visual  : MobileNetV2 + LSTM + Attention")
    print("     Audio   : VGGish (frozen) + Pause + Audio Head")
    print("     Loss    : Weighted BCEWithLogitsLoss")
    print("     Split   : Stratified 80/20")
    print("     Phases  : 4 Sequential (Hard Freeze per Phase)")
    print("     Stopping: Early Stop (patience=3) per Phase")
    print("═" * 70)

    start_time = time.time()

    # ── DEVICE ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")
    os.makedirs(CONFIG['weights_dir'], exist_ok=True)

    # ── COLLECT VIDEOS ───────────────────────────────────────────────────────
    print("\n[1] Collecting Videos...")
    videos, labels = collect_videos()

    if len(videos) == 0:
        print(f"\n  ❌ No videos found!")
        print(f"     Add real videos to : {CONFIG['real_dir']}")
        print(f"     Add scam videos to : {CONFIG['scam_dir']}")
        return

    # ── STRATIFIED SPLIT ─────────────────────────────────────────────────────
    print("\n[2] Stratified 80/20 Split...")
    train_videos, train_labels, val_videos, val_labels = stratified_split(
        videos, labels
    )

    # ── CLASS WEIGHTS ────────────────────────────────────────────────────────
    print("\n[3] Computing Class Weights...")
    pos_weight = compute_pos_weight(train_labels, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── INITIALIZE MODELS ────────────────────────────────────────────────────
    print("\n[4] Initializing Models...")
    visual_model = VisualQualityHead().to(device)
    audio_head   = AudioClassificationHead(
        dropout=CONFIG['audio_dropout']
    ).to(device)
    audio_analyzer = AdvancedAudioAnalyzer(device=str(device))
    print("  ✅ Visual model   : VisualQualityHead")
    print("  ✅ Audio head     : AudioClassificationHead (128→64→1)")
    print("  ✅ Audio analyzer : AdvancedAudioAnalyzer (VGGish frozen)")

    # ── PHASE 1: VISUAL QUALITY HEAD ─────────────────────────────────────────
    phase1_visual_head(
        visual_model,
        train_videos, train_labels,
        val_videos,   val_labels,
        criterion, device
    )

    # ── PHASE 2: VISUAL TEMPORAL ─────────────────────────────────────────────
    phase2_visual_temporal(
        visual_model,
        train_videos, train_labels,
        val_videos,   val_labels,
        criterion, device
    )

    # ── PHASE 3: AUDIO HEAD ──────────────────────────────────────────────────
    phase3_audio_head(
        audio_head, audio_analyzer,
        train_videos, train_labels,
        val_videos,   val_labels,
        criterion, device
    )

    # ── PHASE 4: FUSION ──────────────────────────────────────────────────────
    phase4_fusion(
        visual_model, audio_head, audio_analyzer,
        train_videos, train_labels,
        val_videos,   val_labels,
        criterion, device
    )

    # ── SAVE FINAL MODEL ─────────────────────────────────────────────────────
    save_checkpoint(
        CONFIG['final_model'],
        epoch    = CONFIG['max_epochs_per_phase'],
        val_loss = 0.0,
        val_acc  = 0.0,
        head                  = visual_model.head.state_dict(),
        temporal_lstm         = visual_model.temporal_lstm.state_dict(),
        temporal_attention    = visual_model.temporal_attention.state_dict(),
        temporal_classifier   = visual_model.temporal_classifier.state_dict(),
        fusion_network        = visual_model.fusion_network.state_dict(),
        audio_head            = audio_head.state_dict(),
    )

    # ── TRAINING COMPLETE ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "═" * 70)
    print("  ✅ TRAINING COMPLETE")
    print(f"  Total Time        : {elapsed/60:.1f} minutes")
    print(f"  Best Model        : {CONFIG['best_model']}")
    print(f"  Final Model       : {CONFIG['final_model']}")
    print("\n  Per-Phase Weights :")
    print(f"    Phase 1 (Head)      → {CONFIG['best_visual_head']}")
    print(f"    Phase 2 (Temporal)  → {CONFIG['best_visual_temporal']}")
    print(f"    Phase 3 (Audio)     → {CONFIG['best_audio_head']}")
    print(f"    Phase 4 (Fusion)    → {CONFIG['best_fusion']}")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train()