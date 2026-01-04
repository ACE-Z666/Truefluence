import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os

class VisualQualityHead(nn.Module):
    """
    Visual Quality Engine for TrueFluence.
    
    Role: Analyzes the production quality of the video.
    Input: Video Frames (Images)
    Output: Quality Score (0 = Low Effort/Scammy, 1 = Professional/Credible)
    
    Architecture:
    1. Backbone: MobileNetV2 (Frozen) -> Extracts visual features (lighting, texture, composition).
    2. Head: Custom MLP (Trainable) -> Decides if the features look "Scammy" or "Professional".
    """
    def __init__(self):
        super(VisualQualityHead, self).__init__()
        
        print("Initializing Visual Quality Engine...")
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
            
        # --- PART 2: THE QUALITY ASSESSMENT HEAD (Trainable) ---
        # MobileNetV2 outputs 1280 channels
        print("2. Initializing Custom Quality Assessment Head...")
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),                 # Increased dropout to prevent overfitting on small data
            nn.Linear(1280, 512),              # Compress vector
            nn.ReLU(),
            nn.BatchNorm1d(512),               # Normalize for training stability
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)                  # Output single raw score
            # Note: We removed Sigmoid here if using BCEWithLogitsLoss later, 
            # but if you need a score 0-1 directly, uncomment the line below:
            # , nn.Sigmoid() 
        )

    def forward(self, x):
        """
        Full pass: Image -> Vector -> Quality Score
        """
        # 1. Extract Features (Get the vector)
        # Output shape: (Batch, 1280, 7, 7)
        features = self.backbone(x)
        
        # Global Average Pooling to flatten spatial dims
        # Output shape: (Batch, 1280, 1, 1) -> (Batch, 1280)
        x = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        vector = torch.flatten(x, 1)
        
        # 2. Calculate Quality Score
        score = self.head(vector)
        return score
    
    def get_vector(self, x):
        """
        Helper: Just get the vector representation of the image.
        """
        with torch.no_grad():
            features = self.backbone(x)
            x = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            return torch.flatten(x, 1)

    def save_head_weights(self, path):
        """Save only the trained head weights (small file size)."""
        torch.save(self.head.state_dict(), path)
        print(f"Saved Quality Head weights to {path}")
        
    def load_head_weights(self, path):
        """Load trained head weights."""
        if os.path.exists(path):
            self.head.load_state_dict(torch.load(path))
            print(f"Loaded Quality Head from {path}")
        else:
            print("No trained head found. Using random initialization.")

def extract_quality_frames(video_path, num_frames=10, resize_dim=(224, 224)):
    """
    Extracts frames and prepares them for MobileNetV2 analysis.
    Checks if video can be opened and extracts evenly spaced frames to get a
    good overview of the whole video's quality.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Frame Selection Logic ---
    if total_frames <= 0:
        # Fallback for streams or corrupted headers
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames: raise ValueError(f"No frames in {video_path}")
        
        # Random sample if we can't seek
        indices = sorted(random.sample(range(len(frames)), min(num_frames, len(frames))))
        # Pad if needed
        while len(indices) < num_frames: indices.append(random.choice(indices))
        selected_frames = [frames[i] for i in sorted(indices)]
    else:
        # Smart seeking: Get frames evenly distributed across the video
        # to judge quality of the START, MIDDLE, and END.
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        selected_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret: selected_frames.append(frame)
            else: 
                # If read fails, duplicate last frame
                if selected_frames: selected_frames.append(selected_frames[-1])
        cap.release()

    # --- Preprocessing for MobileNetV2 ---
    # Standard ImageNet normalization constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    processed_frames = []
    for frame in selected_frames:
        if frame is None: continue
        
        # 1. Resize to 224x224 (MobileNet Requirement)
        frame = cv2.resize(frame, resize_dim)
        
        # 2. Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # 4. Normalize with ImageNet stats
        frame = (frame - mean) / std
        
        # 5. Transpose to (Channels, Height, Width) -> PyTorch format
        frame = np.transpose(frame, (2, 0, 1))
        
        processed_frames.append(frame)
        
    if not processed_frames:
         raise ValueError(f"Failed to extract frames from {video_path}")

    # Convert list of arrays to a single Batch Tensor: (Batch_Size, 3, 224, 224)
    tensor_frames = torch.tensor(np.array(processed_frames), dtype=torch.float32)
    
    return tensor_frames