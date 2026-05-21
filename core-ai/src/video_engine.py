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
import pickle

class VisualQualityHead(nn.Module):
    """
    Enhanced Visual Quality Engine for TrueFluence with Complete Multimodal Integration.
    
    Features:
    - Basic quality assessment
    - Temporal consistency analysis  
    - Lighting and background analysis
    - Face detection
    - Multimodal fusion with audio
    - Complete scam detection system
    """
    def __init__(self):
        super(VisualQualityHead, self).__init__()
        
        print("Initializing Complete Enhanced Visual System...")
        print("1. Loading MobileNetV2 Backbone (ImageNet Weights)...")
        
        # Load standard MobileNetV2 with ImageNet weights
        weights = models.MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)
        
        # --- PART 1: THE VECTOR GENERATOR (Frozen Backbone) ---
        self.backbone = base_model.features
        
        # FREEZE the backbone (Resource-constrained hardware)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # --- PART 2: BASIC QUALITY ASSESSMENT HEAD ---
        print("2. Initializing Quality Assessment Head...")
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # --- PART 3: TEMPORAL ANALYSIS COMPONENTS ---
        print("3. Initializing Temporal Analysis...")
        self.temporal_lstm = nn.LSTM(
            input_size    = 1280,
            hidden_size   = 256,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,       # output dim = 256 × 2 = 512
            dropout       = 0.3
        )

        # ── Temporal Attention ─────────────────────────────────────
        # Input must match LSTM output = 512 (bidirectional)
        self.temporal_attention = nn.Sequential(
            nn.Linear(512, 128),        # ← 512 not 256
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # ── Temporal Classifier ────────────────────────────────────
        # Input must match LSTM output = 512 (bidirectional)
        self.temporal_classifier = nn.Sequential(
            nn.Linear(512, 128),        # ← 512 not 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # --- PART 4: MULTIMODAL FUSION NETWORK ---
        print("4. Initializing Multimodal Fusion...")
        self.fusion_network = nn.Sequential(
            nn.Linear(270, 128),  # Combined visual + audio features (135 + 135)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Training status
        self.is_fusion_trained = False
        
        print("✓ Complete Enhanced Visual System Initialized!")

    def forward(self, x):
        """
        Basic forward pass: Image -> Vector -> Quality Score
        """
        features = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        vector = torch.flatten(x, 1)
        score = self.head(vector)
        return score
    
    def forward_temporal(self, frames_batch):
        """
        Temporal LSTM + Attention forward pass.

        Args:
            frames_batch: (B, N, 3, 224, 224)

        Returns:
            score: (B, 1) sigmoid probability
        """
        B, N, C, H, W = frames_batch.shape

        # (B, N, C, H, W) → (B*N, C, H, W)
        x = frames_batch.view(B * N, C, H, W)

        # Backbone (frozen)
        with torch.no_grad():
            feats  = self.backbone(x)
            pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            vecs   = torch.flatten(pooled, 1)                   # (B*N, 1280)

        # (B*N, 1280) → (B, N, 1280)
        vecs = vecs.view(B, N, -1)                              # (B, N, 1280)

        # LSTM
        # Input:  (B, N, 1280)
        # Output: (B, N, 512)  ← 256 hidden × 2 directions
        lstm_out, _ = self.temporal_lstm(vecs)                  # (B, N, 512)

        # Attention
        # Input:  (B, N, 512)
        # Output: (B, N, 1)
        attn_scores  = self.temporal_attention(lstm_out)        # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)        # (B, N, 1)

        # Weighted sum across frames
        # (B, N, 512) × (B, N, 1) → sum → (B, 512)
        context = (lstm_out * attn_weights).sum(dim=1)          # (B, 512)

        # Classifier
        score = self.temporal_classifier(context)               # (B, 1)
        score = torch.sigmoid(score)                            # (B, 1)

        return score
    
    def forward_fusion(self, visual_features, audio_features):
        """
        Multimodal fusion forward pass.
        """
        # Combine visual and audio features
        combined_features = torch.cat([visual_features, audio_features], dim=1)
        
        # Pass through fusion network
        credibility = self.fusion_network(combined_features)
        return credibility
    
    def get_vector(self, x):
        """
        Helper: Just get the vector representation of the image.
        """
        with torch.no_grad():
            features = self.backbone(x)
            x = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            return torch.flatten(x, 1)

    def save_head_weights(self, path):
        """Save all enhanced model weights."""
        torch.save({
            'head': self.head.state_dict(),
            'temporal_encoder': self.temporal_encoder.state_dict(),
            'attention': self.attention.state_dict(),
            'temporal_classifier': self.temporal_classifier.state_dict(),
            'visual_processor': self.visual_processor.state_dict(),
            'audio_processor': self.audio_processor.state_dict(),
            'cross_attention': self.cross_attention.state_dict(),
            'fusion_classifier': self.fusion_classifier.state_dict(),
            'is_fusion_trained': self.is_fusion_trained
        }, path)
        print(f"Saved Complete Enhanced Model to {path}")
        
    def load_head_weights(self, path):
        """Load enhanced model weights with backward compatibility."""
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu')
                
                if isinstance(checkpoint, dict):
                    # Load basic head (always present)
                    if 'head' in checkpoint:
                        self.head.load_state_dict(checkpoint['head'])
                    else:
                        # Old format compatibility
                        self.head.load_state_dict(checkpoint)
                        return
                    
                    # Load enhanced components if available
                    if 'temporal_encoder' in checkpoint:
                        self.temporal_encoder.load_state_dict(checkpoint['temporal_encoder'])
                        self.attention.load_state_dict(checkpoint['attention'])
                        self.temporal_classifier.load_state_dict(checkpoint['temporal_classifier'])
                    
                    if 'visual_processor' in checkpoint:
                        self.visual_processor.load_state_dict(checkpoint['visual_processor'])
                        self.audio_processor.load_state_dict(checkpoint['audio_processor'])
                        self.cross_attention.load_state_dict(checkpoint['cross_attention'])
                        self.fusion_classifier.load_state_dict(checkpoint['fusion_classifier'])
                        self.is_fusion_trained = checkpoint.get('is_fusion_trained', False)
                    
                    print(f"✓ Loaded Enhanced Model from {path}")
                else:
                    # Very old format
                    self.head.load_state_dict(checkpoint)
                    print(f"Loaded Legacy Model from {path}")
                    
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Using random initialization.")
        else:
            print("No trained model found. Using random initialization.")
    
    def extract_comprehensive_visual_features(self, video_path, num_frames=10):
        """
        Extract comprehensive visual features for multimodal analysis.
        """
        try:
            # Extract frames
            frames = extract_quality_frames(video_path, num_frames=num_frames)
            device = next(self.parameters()).device
            frames = frames.to(device)
            
            # Basic quality assessment
            with torch.no_grad():
                quality_scores = self(frames)
                quality_vectors = self.get_vector(frames)
                
                # Average quality metrics
                avg_quality = torch.mean(quality_scores).item()
                avg_vector = torch.mean(quality_vectors, dim=0).cpu().numpy()
            
            # Advanced visual analysis
            lighting_analysis = self.analyze_lighting_consistency(frames)
            background_analysis = self.analyze_background_consistency(frames)
            
            # Temporal analysis
            frames_batch = frames.unsqueeze(0)  # Add batch dimension
            temporal_consistency = self.analyze_temporal_consistency(frames_batch)
            
            # Face detection analysis
            face_analysis_results = []
            for i in range(min(3, len(frames))):
                face_result = self.detect_face_regions(frames[i])
                face_analysis_results.append(face_result['face_area_ratio'])
            
            avg_face_ratio = np.mean(face_analysis_results) if face_analysis_results else 0
            
            # Combine all visual features
            visual_features = np.concatenate([
                avg_vector[:128],  # First 128 MobileNet features
                [avg_quality],
                [lighting_analysis['luminance_consistency']],
                [lighting_analysis['contrast_consistency']],
                [lighting_analysis['overall_lighting_quality']],
                [background_analysis['corner_consistency']],
                [background_analysis['background_stability']],
                [background_analysis['compositing_likelihood']],
                [temporal_consistency['temporal_consistency']],
                [temporal_consistency['consistency_variance']],
                [temporal_consistency['min_similarity']],
                [temporal_consistency['abrupt_changes']],
                [avg_face_ratio]
            ])  # Total: 135 dimensions
            
            return visual_features
            
        except Exception as e:
            print(f"Error extracting visual features: {e}")
            return np.zeros(135)  # Return zero vector on error
    
    def analyze_video_complete(self, video_path, audio_features=None):
        """
        Complete video analysis with all enhanced features.
        """
        print(f"Analyzing video: {video_path}")
        
        # Extract visual features
        visual_features = self.extract_comprehensive_visual_features(video_path)
        
        # Basic visual quality score
        visual_quality = visual_features[128] if len(visual_features) > 128 else 0.5
        
        # If audio features provided, do multimodal fusion
        if audio_features is not None and self.is_fusion_trained:
            device = next(self.parameters()).device
            vis_tensor = torch.FloatTensor(visual_features).unsqueeze(0).to(device)
            aud_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
            
            with torch.no_grad():
                fusion_score = self.forward_fusion(vis_tensor, aud_tensor).item()
        else:
            # Fallback to visual-only analysis
            fusion_score = visual_quality
        
        # Determine verdict
        if fusion_score > 0.7:
            verdict = "REAL / PROFESSIONAL"
            risk_level = "LOW"
        elif fusion_score > 0.4:
            verdict = "UNCERTAIN / MODERATE QUALITY"
            risk_level = "MEDIUM"
        else:
            verdict = "SCAM / LOW EFFORT"
            risk_level = "HIGH"
        
        # Calculate confidence based on feature consistency
        feature_std = np.std(visual_features[:10])  # Consistency of first 10 features
        confidence = max(0.5, min(0.95, 1.0 - feature_std))
        
        return {
            'overall_score': fusion_score,
            'visual_score': visual_quality,
            'confidence': confidence,
            'verdict': verdict,
            'risk_level': risk_level,
            'visual_features': visual_features,
            'has_multimodal': audio_features is not None,
            'fusion_trained': self.is_fusion_trained
        }
    
    def train_fusion_component(self, visual_features_list, audio_features_list, labels, epochs=20, learning_rate=0.001):
        """
        Train the multimodal fusion component.
        """
        print("Training multimodal fusion component...")
        
        device = next(self.parameters()).device
        
        # Convert to tensors
        visual_data = torch.FloatTensor(visual_features_list).to(device)
        audio_data = torch.FloatTensor(audio_features_list).to(device)
        label_data = torch.FloatTensor(labels).unsqueeze(1).to(device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam([
            {'params': self.visual_processor.parameters()},
            {'params': self.audio_processor.parameters()},
            {'params': self.cross_attention.parameters()},
            {'params': self.fusion_classifier.parameters()}
        ], lr=learning_rate)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            predictions = self.forward_fusion(visual_data, audio_data)
            loss = criterion(predictions, label_data)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        self.is_fusion_trained = True
        print("Multimodal fusion training complete!")
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

    def extract_frame_features_enhanced(self, frame_tensor):
        """
        Extract combined features per frame:
        MobileNetV2 + lighting + background folded into one vector
        """
        # 1. MobileNetV2 backbone features
        backbone_features = self.backbone(frame_tensor)  # (1280,)
        
        # 2. Lighting features (folded in)
        frame_np = frame_tensor.permute(1,2,0).cpu().numpy()
        luminance = (0.299 * frame_np[:,:,0] + 
                 0.587 * frame_np[:,:,1] + 
                 0.114 * frame_np[:,:,2])
        
        lighting_feats = torch.tensor([
            luminance.mean(),   # absolute brightness
            luminance.std(),    # contrast level
        ])  # (2,)
        
        # 3. Background corner features (folded in)
        h, w = frame_np.shape[:2]
        corners = [
            frame_np[:30,  :30 ].mean(axis=(0,1)),   # TL
            frame_np[:30,  -30:].mean(axis=(0,1)),   # TR
            frame_np[-30:, :30 ].mean(axis=(0,1)),   # BL
            frame_np[-30:, -30:].mean(axis=(0,1)),   # BR
        ]
        corner_feats = torch.tensor(
            np.concatenate(corners)
        ).float()  # (12,)
        
        # 4. Combine everything into single frame vector
        combined = torch.cat([
            backbone_features,   # 1280
            lighting_feats,      #    2
            corner_feats         #   12
        ])  # Total: 1294-dim per frame
    
        return combined

    def analyze_temporal_consistency(self, video_frames):
        """
        Analyze frame-to-frame consistency for authenticity detection.
        """
        consistency_metrics = {}
        
        # Extract features for all frames
        features = []
        num_frames = video_frames.shape[1] if len(video_frames.shape) == 5 else video_frames.shape[0]
        
        for i in range(num_frames):
            if len(video_frames.shape) == 5:  # (batch, frames, channels, h, w)
                frame = video_frames[0, i]
            else:  # (frames, channels, h, w)
                frame = video_frames[i]
            frame_feat = self.get_vector(frame.unsqueeze(0))
            features.append(frame_feat.cpu().numpy().squeeze())
        
        features = np.array(features)
        
        # Calculate frame-to-frame similarities
        similarities = []
        for i in range(len(features) - 1):
            sim = cosine_similarity([features[i]], [features[i + 1]])[0, 0]
            similarities.append(sim)
        
        consistency_metrics['temporal_consistency'] = np.mean(similarities) if similarities else 1.0
        consistency_metrics['consistency_variance'] = np.var(similarities) if similarities else 0.0
        consistency_metrics['min_similarity'] = np.min(similarities) if similarities else 1.0
        
        # Detect abrupt changes (potential editing artifacts)
        if len(similarities) > 0:
            threshold = np.mean(similarities) - 2 * np.std(similarities)
            abrupt_changes = sum(1 for s in similarities if s < threshold)
            consistency_metrics['abrupt_changes'] = abrupt_changes / len(similarities)
        else:
            consistency_metrics['abrupt_changes'] = 0
        
        return consistency_metrics

# Convenience function for quick analysis
def analyze_video_quick(video_path, audio_features=None, model_path=None, device='cpu'):
    """
    Quick video analysis function using the integrated enhanced system.
    
    Args:
        video_path (str): Path to video file
        audio_features (np.array): Optional audio features for multimodal analysis
        model_path (str): Path to saved model (optional, defaults to enhanced model)
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        dict: Analysis results
    """
    device = torch.device(device)
    model = VisualQualityHead().to(device)
    
    # Default to enhanced model if no path specified
    if model_path is None:
        model_path = os.path.join('models', 'weights', 'enhanced_visual_model.pth')
    
    if model_path and os.path.exists(model_path):
        model.load_head_weights(model_path)
    
    model.eval()
    return model.analyze_video_complete(video_path, audio_features)


# Helper function to create complete scam detector
def create_complete_scam_detector(model_path=None, device='cpu'):
    """
    Create a complete scam detection system.
    
    Args:
        model_path (str): Path to saved model weights
        device (str): Device to use
        
    Returns:
        VisualQualityHead: Complete enhanced scam detector
    """
    device = torch.device(device)
    detector = VisualQualityHead().to(device)
    
    if model_path and os.path.exists(model_path):
        detector.load_head_weights(model_path)
        print(f"✓ Loaded complete detector from {model_path}")
    else:
        print("⚠ Using untrained detector")
    
    return detector

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