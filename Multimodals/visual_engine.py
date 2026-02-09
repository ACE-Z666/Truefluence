import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

class VisualQualityHead(nn.Module):
    """
    Enhanced Visual Quality Engine for TrueFluence.
    
    Role: Analyzes the production quality of the video with sophisticated features.
    Input: Video Frames (Images)
    Output: Quality Score (0 = Low Effort/Scammy, 1 = Professional/Credible)
    
    Architecture:
    1. Backbone: MobileNetV2 (Frozen) -> Extracts visual features (lighting, texture, composition).
    2. Head: Custom MLP (Trainable) -> Decides if the features look "Scammy" or "Professional".
    """
    def __init__(self):
        super(VisualQualityHead, self).__init__()
        
        print("Initializing Enhanced Visual Quality Engine...")
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
        print("2. Initializing Enhanced Quality Assessment Head...")
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),                 # Increased dropout to prevent overfitting on small data
            nn.Linear(1280, 512),              # Compress vector
            nn.ReLU(),
            nn.BatchNorm1d(512),               # Normalize for training stability
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),                 # Output single raw score
            nn.Sigmoid()                       # Output: 0.0 (Scam) to 1.0 (Credible)
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
    
    def analyze_lighting_consistency(self, frames):
        """Analyze lighting patterns across frames for authenticity."""
        lighting_scores = []
        
        for frame in frames:
            # Convert to grayscale for luminance analysis
            if len(frame.shape) == 4:  # Batch dimension
                gray = torch.mean(frame, dim=1)
            else:
                gray = torch.mean(frame, dim=0)
            
            # Calculate lighting metrics
            mean_luminance = torch.mean(gray).item()
            std_luminance = torch.std(gray).item()
            
            lighting_scores.append({
                'mean_luminance': mean_luminance,
                'std_luminance': std_luminance,
                'contrast_ratio': std_luminance / (mean_luminance + 1e-6)
            })
        
        # Analyze consistency across frames
        luminance_values = [score['mean_luminance'] for score in lighting_scores]
        contrast_values = [score['contrast_ratio'] for score in lighting_scores]
        
        consistency_score = {
            'luminance_consistency': 1.0 - np.std(luminance_values) / (np.mean(luminance_values) + 1e-6),
            'contrast_consistency': 1.0 - np.std(contrast_values) / (np.mean(contrast_values) + 1e-6),
            'overall_lighting_quality': np.mean([score['contrast_ratio'] for score in lighting_scores])
        }
        
        return consistency_score
    
    def detect_face_regions(self, frame):
        """Simple face detection using color and texture analysis."""
        # Convert tensor to numpy for OpenCV
        if isinstance(frame, torch.Tensor):
            if len(frame.shape) == 4:  # Batch
                frame = frame[0]
            # Denormalize and convert to uint8
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = ((frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255).astype(np.uint8)
        
        # Simple skin color detection (approximation)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_area = np.sum(skin_mask > 0) / (frame.shape[0] * frame.shape[1])
        
        return {
            'has_face': skin_area > 0.02,  # At least 2% skin-colored pixels
            'face_area_ratio': skin_area,
            'skin_mask': skin_mask
        }
    
    def analyze_background_consistency(self, frames):
        """Analyze background consistency to detect compositing artifacts."""
        background_features = []
        
        for frame in frames:
            # Extract features from corners (likely background regions)
            corners = {
                'top_left': frame[..., :50, :50],
                'top_right': frame[..., :50, -50:],
                'bottom_left': frame[..., -50:, :50],
                'bottom_right': frame[..., -50:, -50:]
            }
            
            corner_features = {}
            for corner_name, corner_region in corners.items():
                corner_features[corner_name] = {
                    'mean_color': torch.mean(corner_region, dim=(-2, -1)).cpu().numpy(),
                    'texture_variance': torch.var(corner_region, dim=(-2, -1)).cpu().numpy()
                }
            
            background_features.append(corner_features)
        
        # Calculate consistency across frames
        consistency_scores = []
        for corner_name in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            corner_means = [frame[corner_name]['mean_color'] for frame in background_features]
            corner_variances = [frame[corner_name]['texture_variance'] for frame in background_features]
            
            # Calculate consistency
            mean_consistency = 1.0 - np.std(corner_means, axis=0).mean() / (np.mean(corner_means, axis=0).mean() + 1e-6)
            var_consistency = 1.0 - np.std(corner_variances, axis=0).mean() / (np.mean(corner_variances, axis=0).mean() + 1e-6)
            
            consistency_scores.append((mean_consistency + var_consistency) / 2)
        
        return {
            'corner_consistency': np.mean(consistency_scores),
            'background_stability': np.min(consistency_scores),
            'compositing_likelihood': 1.0 - np.mean(consistency_scores)
        }

class TemporalVisualAnalyzer(nn.Module):
    """
    Temporal analysis for video sequences to detect inconsistencies and patterns.
    """
    def __init__(self, feature_dim=1280, hidden_dim=256):
        super(TemporalVisualAnalyzer, self).__init__()
        
        print("Initializing Temporal Visual Analyzer...")
        
        # Visual feature extractor (frozen MobileNetV2)
        self.visual_head = VisualQualityHead()
        
        # Temporal analysis components
        self.temporal_encoder = nn.LSTM(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, video_frames):
        """
        Analyze temporal patterns in video frames.
        Args:
            video_frames: Tensor of shape (batch_size, num_frames, channels, height, width)
        """
        batch_size, num_frames = video_frames.shape[0], video_frames.shape[1]
        
        # Extract features from all frames
        frame_features = []
        for i in range(num_frames):
            frame = video_frames[:, i]  # (batch_size, channels, height, width)
            features = self.visual_head.get_vector(frame)  # (batch_size, 1280)
            frame_features.append(features)
        
        # Stack features: (batch_size, num_frames, 1280)
        sequence_features = torch.stack(frame_features, dim=1)
        
        # Temporal encoding with LSTM
        lstm_out, _ = self.temporal_encoder(sequence_features)
        
        # Self-attention for important temporal patterns
        attended_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output for classification
        final_features = attended_features[:, -1, :]
        
        # Classification
        credibility_score = self.classifier(final_features)
        
        return credibility_score
    
    def analyze_temporal_consistency(self, video_frames):
        """
        Analyze frame-to-frame consistency for authenticity detection.
        """
        consistency_metrics = {}
        
        # Extract features for all frames
        features = []
        for i in range(video_frames.shape[1]):
            frame_feat = self.visual_head.get_vector(video_frames[:, i])
            features.append(frame_feat.cpu().numpy())
        
        features = np.array(features).squeeze()
        
        # Calculate frame-to-frame similarities
        similarities = []
        for i in range(len(features) - 1):
            sim = cosine_similarity([features[i]], [features[i + 1]])[0, 0]
            similarities.append(sim)
        
        consistency_metrics['temporal_consistency'] = np.mean(similarities)
        consistency_metrics['consistency_variance'] = np.var(similarities)
        consistency_metrics['min_similarity'] = np.min(similarities)
        
        # Detect abrupt changes (potential editing artifacts)
        threshold = np.mean(similarities) - 2 * np.std(similarities)
        abrupt_changes = sum(1 for s in similarities if s < threshold)
        consistency_metrics['abrupt_changes'] = abrupt_changes / len(similarities)
        
        return consistency_metrics
    
    def save_weights(self, path):
        """Save temporal analyzer weights."""
        torch.save({
            'temporal_encoder': self.temporal_encoder.state_dict(),
            'attention': self.attention.state_dict(),
            'classifier': self.classifier.state_dict()
        }, path)
        print(f"Saved Temporal Analyzer weights to {path}")
        
    def load_weights(self, path):
        """Load temporal analyzer weights."""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.temporal_encoder.load_state_dict(checkpoint['temporal_encoder'])
            self.attention.load_state_dict(checkpoint['attention'])
            self.classifier.load_state_dict(checkpoint['classifier'])
            print(f"Loaded Temporal Analyzer from {path}")
        else:
            print("No temporal weights found. Using random initialization.")

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