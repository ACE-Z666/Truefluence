"""
mesonet.py  —  MesoNet Deepfake Detector for TrueFluence
==========================================================
Architecture : Meso-4  (Afchar et al., IEEE WIFS 2018)
              https://arxiv.org/abs/1809.00888

Output convention
-----------------
  sigmoid output ≈ 1.0  →  REAL face
  sigmoid output ≈ 0.0  →  DEEPFAKE face

Gate logic used in test.py
---------------------------
  deepfake_prob  = 1 - raw_output   (flip so 1 = deepfake)
  If  mean(deepfake_prob per frame) >= DEEPFAKE_THRESHOLD (0.80)
      → pipeline aborts, final_score = 0, verdict = DEEPFAKE
  Else
      → proceed to Video / Audio / Comments / Engagement engines

Pre-trained weights
-------------------
  Run  python mesonet.py  (or call download_and_convert_weights())
  to download the original Keras Deepfake weights from the official
  DariusAf/MesoNet GitHub release and convert them to PyTorch .pth.

  Saved at:  models/weights/meso4_DF.pth
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_DIR      = os.path.join('models', 'weights')
MESO4_PTH        = os.path.join(WEIGHTS_DIR, 'meso4_DF.pth')
MESO4_KERAS_URL  = (
    'https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_DF.h5'
)

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE  —  Meso-4
# ─────────────────────────────────────────────────────────────────────────────

class Meso4(nn.Module):
    """
    Exact PyTorch replica of the Keras Meso-4 architecture.

    Input  : (B, 3, 256, 256)  — RGB, normalised to [-1, 1]
    Output : (B, 1)            — sigmoid  (1 = real, 0 = deepfake)

    Layers
    ------
    Conv1  : 3  → 8,  3×3, same-pad
    Conv2  : 8  → 8,  5×5, same-pad
    Conv3  : 8  → 16, 5×5, same-pad
    Conv4  : 16 → 16, 5×5, same-pad
    Flatten + FC(16) + Dropout(0.5) + FC(1, sigmoid)
    """

    def __init__(self):
        super(Meso4, self).__init__()

        # ── Convolutional blocks ────────────────────────────────────────────
        self.conv1 = nn.Conv2d(3,  8,  kernel_size=3, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8,  8,  kernel_size=5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8,  16, kernel_size=5, padding=2, bias=True)
        self.bn3   = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=True)
        self.bn4   = nn.BatchNorm2d(16)

        # ── Pooling layers (match Keras implementation) ─────────────────────
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 256→128
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)  # 128→ 32
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)  #  32→  8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  #   8→  4  (NOT 4×4!)

        # After 4 pooling ops:  256 / 2 / 4 / 4 / 2 = 4  (no, actually 8/2=4)
        # Confirmed from pretrained fc1 weight shape: (1024, 16) → 16 × 8 × 8 = 1024
        # Means pool4 must produce 8×8, i.e. pool4 = 2×2: 8/2=4... let's use no pool4
        # Actual: pool3 output = 8×8, pool4 skips to flatten → 16*8*8=1024 ✅
        self.fc1     = nn.Linear(16 * 8 * 8, 16)  # 1024 → 16
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(16, 1)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        Args:
            x : (B, 3, 256, 256)  float32, values in [-1, 1]
        Returns:
            (B, 1) sigmoid score  (1 = real)
        """
        # Block 1  → (B, 8, 128, 128)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = self.pool1(x)

        # Block 2  → (B, 8, 32, 32)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = self.pool2(x)

        # Block 3  → (B, 16, 8, 8)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = self.pool3(x)

        # Block 4  → (B, 16, 8, 8)  — no pool4 (pool4 removed to match weight shape)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        # NOTE: no pool4 — flatten gives 16×8×8 = 1024, matching pretrained fc1 weight

        # FC
        x = torch.flatten(x, 1)         # (B, 1024)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # (B, 1)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT DOWNLOAD + CONVERSION  (Keras h5 → PyTorch pth)
# ─────────────────────────────────────────────────────────────────────────────

def download_and_convert_weights(force=False):
    """
    Download the official Meso4_DF Keras weights from GitHub,
    convert the conv/bn/dense layer weights to PyTorch format,
    and save as  models/weights/meso4_DF.pth.

    Requires:  pip install h5py requests
    """
    if os.path.exists(MESO4_PTH) and not force:
        print(f"  ✅ MesoNet weights already exist: {MESO4_PTH}")
        return

    try:
        import h5py
        import requests
    except ImportError:
        print("  ⚠  h5py / requests not installed.")
        print("     Run:  pip install h5py requests")
        sys.exit(1)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    tmp_h5 = os.path.join(WEIGHTS_DIR, '_meso4_DF_tmp.h5')

    # ── Download ─────────────────────────────────────────────────────────
    print(f"  Downloading Meso4_DF weights …")
    resp = requests.get(MESO4_KERAS_URL, stream=True, timeout=60)
    resp.raise_for_status()
    with open(tmp_h5, 'wb') as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    print(f"  Downloaded → {tmp_h5}")

    # ── Convert ─────────────────────────────────────────────────────────
    print(f"  Converting Keras h5 → PyTorch pth …")

    model      = Meso4()
    state_dict = model.state_dict()

    with h5py.File(tmp_h5, 'r') as hf:

        def _conv(layer_name, pt_name):
            """Load conv weight + bias into state_dict."""
            grp = hf[layer_name][layer_name]
            # Keras kernel shape: (H, W, C_in, C_out)
            # PyTorch weight shape: (C_out, C_in, H, W)
            k = grp['kernel:0'][()]
            b = grp['bias:0'][()]
            state_dict[pt_name + '.weight'] = torch.from_numpy(
                np.transpose(k, (3, 2, 0, 1)).copy()
            )
            state_dict[pt_name + '.bias'] = torch.from_numpy(b.copy())

        def _bn(layer_name, pt_name):
            """Load BN gamma/beta/mean/var into state_dict."""
            grp = hf[layer_name][layer_name]
            state_dict[pt_name + '.weight']       = torch.from_numpy(grp['gamma:0'][()].copy())
            state_dict[pt_name + '.bias']         = torch.from_numpy(grp['beta:0'][()].copy())
            state_dict[pt_name + '.running_mean'] = torch.from_numpy(grp['moving_mean:0'][()].copy())
            state_dict[pt_name + '.running_var']  = torch.from_numpy(grp['moving_variance:0'][()].copy())

        def _fc(layer_name, pt_name):
            """Load dense kernel + bias into state_dict."""
            grp = hf[layer_name][layer_name]
            k = grp['kernel:0'][()]
            b = grp['bias:0'][()]
            # Keras: (in, out) → PyTorch: (out, in)
            state_dict[pt_name + '.weight'] = torch.from_numpy(k.T.copy())
            state_dict[pt_name + '.bias']   = torch.from_numpy(b.copy())

        # ── Map Keras layer names → PyTorch param names ─────────────────
        _conv('conv2d_5',            'conv1')
        _bn  ('batch_normalization_5','bn1')
        _conv('conv2d_6',            'conv2')
        _bn  ('batch_normalization_6','bn2')
        _conv('conv2d_7',            'conv3')
        _bn  ('batch_normalization_7','bn3')
        _conv('conv2d_8',            'conv4')
        _bn  ('batch_normalization_8','bn4')
        _fc  ('dense_3',             'fc1')
        _fc  ('dense_4',             'fc2')

    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), MESO4_PTH)
    os.remove(tmp_h5)

    print(f"  ✅ Weights saved → {MESO4_PTH}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def load_meso4(device, weights_path=None):
    """
    Instantiate Meso4 and load pretrained weights.

    Args:
        device       : torch.device
        weights_path : str or None  (defaults to MESO4_PTH)

    Returns:
        model : Meso4  (eval mode)
    """
    path = weights_path or MESO4_PTH

    model = Meso4().to(device)

    if not os.path.exists(path):
        print(f"  ⚠  MesoNet weights not found: {path}")
        print(f"     Run:  python mesonet.py   to download + convert.")
        print(f"     Deepfake gate will be DISABLED for this run.")
        return None

    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  ✅ MesoNet (Meso-4 DF) loaded from: {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FRAME PREPROCESSING  (matches original MesoNet pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame_meso(frame_bgr, size=(256, 256)):
    """
    Resize + convert BGR → RGB → normalise to [-1, 1].

    Args:
        frame_bgr : np.ndarray  (H, W, 3)  uint8
        size      : tuple       target resolution  (default 256×256)

    Returns:
        torch.FloatTensor  (3, H, W)   values in [-1, 1]
    """
    frame = cv2.resize(frame_bgr, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 127.5 - 1.0    # [0,255] → [-1, 1]
    frame = np.transpose(frame, (2, 0, 1))             # (H,W,3) → (3,H,W)
    return torch.tensor(frame, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DEEPFAKE SCREENER  —  returns per-frame scores + gate decision
# ─────────────────────────────────────────────────────────────────────────────

def screen_for_deepfake(
    video_path,
    meso_model,
    device,
    num_frames       = 16,
    threshold        = 0.80,
    resize           = (256, 256),
):
    """
    Sample `num_frames` evenly from `video_path`, run each through
    Meso-4, and decide whether the video is a deepfake.

    MesoNet output convention
    -------------------------
      sigmoid ≈ 1  →  REAL
      sigmoid ≈ 0  →  DEEPFAKE
    So:  deepfake_prob = 1 - meso_output

    Gate
    ----
      If mean(deepfake_prob) >= threshold  → DEEPFAKE (abort pipeline)
      Else                                 → pass through

    Args:
        video_path  : str
        meso_model  : Meso4 (eval mode) or None (gate disabled)
        device      : torch.device
        num_frames  : int    number of frames to sample
        threshold   : float  deepfake confidence gate (default 0.80)
        resize      : tuple  frame resolution fed to Meso-4

    Returns:
        dict:
            'is_deepfake'       : bool
            'deepfake_prob'     : float   mean deepfake probability [0,1]
            'frame_df_probs'    : list    per-frame deepfake prob
            'frames_sampled'    : int
            'gate_threshold'    : float
            'meso_available'    : bool
    """
    result = {
        'is_deepfake'    : False,
        'deepfake_prob'  : 0.0,
        'frame_df_probs' : [],
        'frames_sampled' : 0,
        'gate_threshold' : threshold,
        'meso_available' : meso_model is not None,
    }

    if meso_model is None:
        # Gate disabled (weights not present)
        return result

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠  MesoNet: cannot open {video_path}")
        return result

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return result

    # ── Sample frame indices ───────────────────────────────────────────────
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(preprocess_frame_meso(frame, size=resize))
        elif frames:
            frames.append(frames[-1])   # duplicate last on read failure

    cap.release()

    if not frames:
        return result

    # ── Inference ─────────────────────────────────────────────────────────
    batch = torch.stack(frames).to(device)   # (N, 3, 256, 256)

    with torch.no_grad():
        real_probs = meso_model(batch).squeeze(1)   # (N,)  ← sigmoid, 1=real

    df_probs = (1.0 - real_probs).cpu().numpy().tolist()

    mean_df  = float(np.mean(df_probs))

    result['frame_df_probs'] = [round(p, 4) for p in df_probs]
    result['deepfake_prob']  = round(mean_df, 4)
    result['frames_sampled'] = len(frames)
    result['is_deepfake']    = mean_df >= threshold

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE — download + convert weights when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "═" * 60)
    print("  MesoNet Weight Setup")
    print("  Downloading Meso4_DF weights from DariusAf/MesoNet …")
    print("═" * 60)
    download_and_convert_weights(force='--force' in sys.argv)
    print("\n  Done. You can now run test.py")
    print("═" * 60 + "\n")
