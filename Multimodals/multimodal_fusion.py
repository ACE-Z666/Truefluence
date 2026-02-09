import torch
import torch.nn as nn
import numpy as np
from visual_engine import VisualQualityHead, TemporalVisualAnalyzer, extract_quality_frames
from audio_engine import AudioFeatureExtractor, AudioScamDetector, AdvancedAudioAnalyzer
import pickle
import os

class MultimodalFusionNetwork(nn.Module):
    """
    Fusion network that combines visual and audio features for final credibility assessment.
    """
    def __init__(self, visual_dim=135, audio_dim=135, hidden_dim=256):
        super(MultimodalFusionNetwork, self).__init__()
        
        print("Initializing Multimodal Fusion Network...")
        
        # Feature processing layers
        self.visual_processor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.audio_processor = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features, audio_features):
        """
        Forward pass through fusion network.
        
        Args:
            visual_features: Tensor of visual features
            audio_features: Tensor of audio features
            
        Returns:
            Tensor: Credibility score (0-1)
        """
        # Process features through modality-specific layers
        vis_processed = self.visual_processor(visual_features)
        aud_processed = self.audio_processor(audio_features)
        
        # Add sequence dimension for attention
        vis_seq = vis_processed.unsqueeze(1)  # (batch, 1, hidden_dim//2)
        aud_seq = aud_processed.unsqueeze(1)  # (batch, 1, hidden_dim//2)
        
        # Cross-modal attention (visual attending to audio)
        vis_attended, _ = self.cross_attention(vis_seq, aud_seq, aud_seq)
        aud_attended, _ = self.cross_attention(aud_seq, vis_seq, vis_seq)
        
        # Combine attended features
        combined = torch.cat([
            vis_attended.squeeze(1), 
            aud_attended.squeeze(1)
        ], dim=1)
        
        # Final prediction
        credibility = self.fusion_layer(combined)
        return credibility

class MultimodalScamDetector:
    """
    Comprehensive multimodal scam detection system combining visual and audio analysis.
    """
    def __init__(self, device='cpu'):
        self.device = device
        
        print("Initializing Multimodal Scam Detection System...")
        
        # Initialize individual analyzers
        self.visual_analyzer = VisualQualityHead().to(device)
        self.temporal_analyzer = TemporalVisualAnalyzer().to(device)
        self.audio_analyzer = AdvancedAudioAnalyzer(device)
        
        # Initialize fusion network
        self.fusion_network = MultimodalFusionNetwork().to(device)
        
        # Training status
        self.is_trained = False
        
    def extract_visual_features(self, video_path, num_frames=10):
        """
        Extract comprehensive visual features from video.
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to analyze
            
        Returns:
            np.ndarray: Combined visual feature vector
        """
        try:
            # Extract frames
            frames = extract_quality_frames(video_path, num_frames=num_frames)
            frames = frames.to(self.device)
            
            # Basic quality assessment
            with torch.no_grad():
                quality_scores = self.visual_analyzer(frames)
                quality_vectors = self.visual_analyzer.get_vector(frames)
                
                # Average quality metrics
                avg_quality = torch.mean(quality_scores).item()
                avg_vector = torch.mean(quality_vectors, dim=0).cpu().numpy()
            
            # Advanced visual analysis
            lighting_analysis = self.visual_analyzer.analyze_lighting_consistency(frames)
            background_analysis = self.visual_analyzer.analyze_background_consistency(frames)
            
            # Temporal analysis
            frames_batch = frames.unsqueeze(0)  # Add batch dimension
            temporal_consistency = self.temporal_analyzer.analyze_temporal_consistency(frames_batch)
            
            # Face detection analysis
            face_analysis_results = []
            for i in range(min(3, len(frames))):  # Analyze first 3 frames for faces
                face_result = self.visual_analyzer.detect_face_regions(frames[i])
                face_analysis_results.append(face_result['face_area_ratio'])
            
            avg_face_ratio = np.mean(face_analysis_results) if face_analysis_results else 0
            
            # Combine all visual features
            visual_features = np.concatenate([
                avg_vector,  # 1280 dimensions from MobileNetV2
                [avg_quality],  # 1 dimension
                [lighting_analysis['luminance_consistency']],  # 1 dimension
                [lighting_analysis['contrast_consistency']],  # 1 dimension
                [lighting_analysis['overall_lighting_quality']],  # 1 dimension
                [background_analysis['corner_consistency']],  # 1 dimension
                [background_analysis['background_stability']],  # 1 dimension
                [background_analysis['compositing_likelihood']],  # 1 dimension
                [temporal_consistency['temporal_consistency']],  # 1 dimension
                [temporal_consistency['consistency_variance']],  # 1 dimension
                [temporal_consistency['min_similarity']],  # 1 dimension
                [temporal_consistency['abrupt_changes']],  # 1 dimension
                [avg_face_ratio]  # 1 dimension
            ])  # Total: 1292 dimensions
            
            # Reduce to target dimension (135) for fusion
            # Use PCA-like reduction by selecting most important features
            important_indices = np.concatenate([
                np.arange(0, 128),  # First 128 MobileNet features
                np.arange(1280, visual_features.shape[0])  # All additional features
            ])
            
            return visual_features[important_indices]
            
        except Exception as e:
            print(f"Error extracting visual features: {e}")
            return np.zeros(135)  # Return zero vector on error
    
    def analyze_video(self, video_path):
        """
        Comprehensive video analysis using all modalities.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Comprehensive analysis results
        """
        print(f"Analyzing video: {video_path}")
        
        # Extract features from both modalities
        visual_features = self.extract_visual_features(video_path)
        audio_features = self.audio_analyzer.get_comprehensive_audio_features(video_path)
        
        # Individual modality scores
        visual_quality = np.mean(visual_features[:5]) if len(visual_features) > 5 else 0.5
        audio_quality = self.audio_analyzer.analyze_voice_authenticity(video_path)['authenticity_score']
        
        # Fusion analysis
        if self.is_trained:
            with torch.no_grad():
                vis_tensor = torch.FloatTensor(visual_features).unsqueeze(0).to(self.device)
                aud_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
                
                fusion_score = self.fusion_network(vis_tensor, aud_tensor).item()
        else:
            # Fallback: weighted average if not trained
            fusion_score = 0.6 * visual_quality + 0.4 * audio_quality
        
        # Detailed analysis results
        voice_analysis = self.audio_analyzer.analyze_voice_authenticity(video_path)
        emotion_analysis = self.audio_analyzer.analyze_emotional_patterns(video_path)
        
        # Confidence calculation
        consistency_score = (
            voice_analysis['voice_consistency'] + 
            emotion_analysis['emotional_authenticity']
        ) / 2
        
        confidence = min(0.95, max(0.05, consistency_score))
        
        # Final verdict
        if fusion_score > 0.7:
            verdict = "REAL / PROFESSIONAL"
            risk_level = "LOW"
        elif fusion_score > 0.4:
            verdict = "UNCERTAIN / MODERATE QUALITY"
            risk_level = "MEDIUM"
        else:
            verdict = "SCAM / LOW EFFORT"
            risk_level = "HIGH"
        
        return {
            'overall_score': fusion_score,
            'confidence': confidence,
            'verdict': verdict,
            'risk_level': risk_level,
            'visual_score': visual_quality,
            'audio_score': audio_quality,
            'voice_analysis': voice_analysis,
            'emotion_analysis': emotion_analysis,
            'detailed_scores': {
                'visual_quality': visual_quality,
                'audio_authenticity': audio_quality,
                'voice_consistency': voice_analysis['voice_consistency'],
                'emotional_authenticity': emotion_analysis['emotional_authenticity'],
                'speech_naturalness': emotion_analysis['speech_naturalness']
            }
        }
    
    def train_fusion_network(self, video_paths, labels, epochs=20, learning_rate=0.001):
        """
        Train the fusion network on labeled data.
        
        Args:
            video_paths (list): List of video file paths
            labels (list): List of labels (0 for scam, 1 for real)
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
        """
        print("Training fusion network...")
        
        # Extract features for all videos
        visual_features_list = []
        audio_features_list = []
        
        for video_path in video_paths:
            print(f"Processing {video_path}...")
            vis_feat = self.extract_visual_features(video_path)
            aud_feat = self.audio_analyzer.get_comprehensive_audio_features(video_path)
            
            visual_features_list.append(vis_feat)
            audio_features_list.append(aud_feat)
        
        # Convert to tensors
        visual_data = torch.FloatTensor(visual_features_list).to(self.device)
        audio_data = torch.FloatTensor(audio_features_list).to(self.device)
        label_data = torch.FloatTensor(labels).unsqueeze(1).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.fusion_network.parameters(), lr=learning_rate)
        
        # Training loop
        self.fusion_network.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            predictions = self.fusion_network(visual_data, audio_data)
            loss = criterion(predictions, label_data)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print("Fusion network training complete!")
    
    def save_models(self, base_path):
        """
        Save all trained models.
        
        Args:
            base_path (str): Base directory to save models
        """
        os.makedirs(base_path, exist_ok=True)
        
        # Save visual model
        visual_path = os.path.join(base_path, "visual_quality_head.pth")
        self.visual_analyzer.save_head_weights(visual_path)
        
        # Save temporal model
        temporal_path = os.path.join(base_path, "temporal_analyzer.pth")
        self.temporal_analyzer.save_weights(temporal_path)
        
        # Save fusion network
        fusion_path = os.path.join(base_path, "fusion_network.pth")
        torch.save(self.fusion_network.state_dict(), fusion_path)
        
        print(f"All models saved to {base_path}")
    
    def load_models(self, base_path):
        """
        Load all trained models.
        
        Args:
            base_path (str): Base directory containing saved models
        """
        # Load visual model
        visual_path = os.path.join(base_path, "visual_quality_head.pth")
        if os.path.exists(visual_path):
            self.visual_analyzer.load_head_weights(visual_path)
        
        # Load temporal model
        temporal_path = os.path.join(base_path, "temporal_analyzer.pth")
        if os.path.exists(temporal_path):
            self.temporal_analyzer.load_weights(temporal_path)
        
        # Load fusion network
        fusion_path = os.path.join(base_path, "fusion_network.pth")
        if os.path.exists(fusion_path):
            self.fusion_network.load_state_dict(torch.load(fusion_path))
            self.is_trained = True
            print("All models loaded successfully!")
        else:
            print("Fusion network not found. Using untrained network.")

# Convenience function for quick analysis
def analyze_video_quick(video_path, model_path=None, device='cpu'):
    """
    Quick video analysis function.
    
    Args:
        video_path (str): Path to video file
        model_path (str): Path to saved models (optional)
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        dict: Analysis results
    """
    detector = MultimodalScamDetector(device=device)
    
    if model_path and os.path.exists(model_path):
        detector.load_models(model_path)
    
    return detector.analyze_video(video_path)