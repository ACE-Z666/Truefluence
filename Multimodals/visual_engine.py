import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os

class VisualDeepfakeDetector(nn.Module):
    """
    Visual Engine for TrueFluence.
    
    Architecture:
    1. Backbone: MobileNetV2 (Frozen) -> Converts Image to 1280-dim Vector.
    2. Head: Custom MLP (Trainable) -> Converts Vector to Credibility Score.
    """
    def __init__(self):
        super(VisualDeepfakeDetector, self).__init__()
        
        print("Initializing Visual Engine...")
        print("1. Loading MobileNetV2 Backbone (ImageNet Weights)...")
        
        # Load standard MobileNetV2 with ImageNet weights
        weights = models.MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)
        
        # --- PART 1: THE VECTOR GENERATOR (Frozen Backbone) ---
        # We only keep the feature extraction layers
        self.backbone = base_model.features
        
        # FREEZE the backbone (Constraint: Resource-constrained hardware)
        # We do not train these layers. They just convert pixels to vectors.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # --- PART 2: THE CREDIBILITY HEAD (Trainable) ---
        # MobileNetV2 outputs 1280 channels
        print("2. Initializing Custom Credibility Head...")
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 256),      # Compress vector
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),         # Output single score
            nn.Sigmoid()               # Normalize to 0-1 (Probability)
        )

    def forward(self, x):
        """
        Full pass: Image -> Vector -> Score
        """
        # 1. Extract Features (Get the vector)
        # Output shape: (Batch, 1280, 7, 7)
        features = self.backbone(x)
        
        # Global Average Pooling to flatten spatial dims
        # Output shape: (Batch, 1280, 1, 1) -> (Batch, 1280)
        x = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        vector = torch.flatten(x, 1)
        
        # 2. Calculate Score
        score = self.head(vector)
        return score
    
    def get_vector(self, x):
        """
        Helper: Just get the vector representation of the image (for debugging or analysis).
        """
        with torch.no_grad():
            features = self.backbone(x)
            x = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            return torch.flatten(x, 1)

    def save_head_weights(self, path):
        """Save only the trained head weights (small file size)."""
        torch.save(self.head.state_dict(), path)
        
    def load_head_weights(self, path):
        """Load trained head weights."""
        if os.path.exists(path):
            self.head.load_state_dict(torch.load(path))
            print(f"Loaded trained head from {path}")
        else:
            print("No trained head found. Using random initialization.")

def extract_frames(video_path, num_frames=10, resize_dim=(224, 224)):
    """
    Extracts frames and prepares them for MobileNetV2.
    
    Changes from MesoNet:
    1. Resize to 224x224 (Standard for MobileNet).
    2. Normalize using ImageNet Mean/Std (Required for pre-trained weights).
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Frame Selection Logic ---
    if total_frames <= 0:
        # Fallback for streams
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames: raise ValueError(f"No frames in {video_path}")
        
        indices = sorted(random.sample(range(len(frames)), min(num_frames, len(frames))))
        # Pad if needed
        while len(indices) < num_frames: indices.append(random.choice(indices))
        selected_frames = [frames[i] for i in sorted(indices)]
    else:
        # Efficient seeking
        indices = sorted(random.sample(range(total_frames), min(num_frames, total_frames)))
        while len(indices) < num_frames: indices.append(random.choice(indices))
        indices.sort()
        
        selected_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret: selected_frames.append(frame)
            else: 
                if selected_frames: selected_frames.append(selected_frames[-1])
        cap.release()

    # --- Preprocessing for MobileNetV2 ---
    # Standard ImageNet normalization constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    processed_frames = []
    for frame in selected_frames:
        # 1. Resize to 224x224
        frame = cv2.resize(frame, resize_dim)
        
        # 2. Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # 4. Normalize with ImageNet stats (Important for MobileNet!)
        frame = (frame - mean) / std
        
        # 5. Transpose to (Channels, Height, Width)
        frame = np.transpose(frame, (2, 0, 1))
        
        processed_frames.append(frame)
        
    if not processed_frames:
         raise ValueError(f"Failed to extract frames from {video_path}")

    tensor_frames = torch.tensor(np.array(processed_frames), dtype=torch.float32)
    
    return tensor_frames
