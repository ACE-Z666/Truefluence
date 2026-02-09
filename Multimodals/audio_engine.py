import torch
import numpy as np
import librosa
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.signal import spectrogram
import pickle

class AudioFeatureExtractor:
    """
    Extracts audio features using a frozen VGGish backbone.
    """
    def __init__(self, device='cpu'):
        self.device = device
        print("Loading VGGish model...")
        # Load VGGish from torch.hub
        # Note: This requires internet access on first run to download the model
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval() # Frozen Backbone
        self.model.to(self.device)
        
    def preprocess_audio(self, y, sr):
        """
        Preprocesses audio to Log Mel Spectrograms compatible with VGGish.
        VGGish expects: 96 frames x 64 mel bands.
        """
        # VGGish parameters
        SAMPLE_RATE = 16000
        N_FFT = 400      # 25ms at 16kHz
        HOP_LENGTH = 160 # 10ms at 16kHz
        N_MELS = 64
        
        if sr != SAMPLE_RATE:
             raise ValueError("Audio must be 16kHz")

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            n_mels=N_MELS
        )
        
        # Convert to Log Mel Spectrogram
        # Add small constant to avoid log(0)
        log_mel_spec = np.log(mel_spec + 1e-6)
        
        # Transpose to (Time, Mel) -> (T, 64)
        log_mel_spec = log_mel_spec.T
        
        # VGGish expects input shape (N, 1, 96, 64)
        # We need to frame the audio into 0.96s chunks (96 frames)
        # For this MVP, we'll take the center 96 frames or pad if too short
        
        num_frames = log_mel_spec.shape[0]
        target_frames = 96
        
        if num_frames < target_frames:
            # Pad with zeros
            padding = target_frames - num_frames
            log_mel_spec = np.pad(log_mel_spec, ((0, padding), (0, 0)), mode='constant')
        elif num_frames > target_frames:
            # Take center crop
            start = (num_frames - target_frames) // 2
            log_mel_spec = log_mel_spec[start:start+target_frames, :]
            
        # Add batch and channel dimensions: (1, 1, 96, 64)
        input_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return input_tensor

    def process_audio(self, video_path):
        """
        Extracts audio from video, resamples to 16kHz, and generates embeddings.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            numpy.ndarray: 128-dimensional embedding vector.
        """
        try:
            # Load audio using librosa
            y, sr = librosa.load(video_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Error loading audio from {video_path}: {e}")
            return np.zeros(128)

        if len(y) == 0:
             return np.zeros(128)

        try:
            # Preprocess
            input_tensor = self.preprocess_audio(y, sr)
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                # Forward pass
                embedding = self.model(input_tensor)
                
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error processing audio features: {e}")
            return np.zeros(128)

    def extract_temporal_features(self, y, sr, chunk_duration=2.0):
        """
        Extract temporal audio features for consistency analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate  
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            dict: Temporal features including consistency metrics
        """
        chunk_samples = int(chunk_duration * sr)
        num_chunks = len(y) // chunk_samples
        
        if num_chunks < 2:
            return {'temporal_consistency': 0.5, 'energy_variance': 0.5, 'pitch_stability': 0.5}
        
        chunk_features = []
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples
            chunk = y[start_idx:end_idx]
            
            # Extract features for each chunk
            energy = np.mean(chunk ** 2)
            zcr = librosa.feature.zero_crossing_rate(chunk)[0].mean()
            spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0].mean()
            
            chunk_features.append([energy, zcr, spectral_centroid])
        
        chunk_features = np.array(chunk_features)
        
        # Calculate consistency metrics
        consistency_scores = []
        for feature_idx in range(chunk_features.shape[1]):
            feature_values = chunk_features[:, feature_idx]
            consistency = 1.0 - (np.std(feature_values) / (np.mean(feature_values) + 1e-6))
            consistency_scores.append(max(0, min(1, consistency)))
        
        return {
            'temporal_consistency': np.mean(consistency_scores),
            'energy_variance': np.var(chunk_features[:, 0]),
            'pitch_stability': consistency_scores[2] if len(consistency_scores) > 2 else 0.5
        }

class AudioScamDetector:
    """
    Wrapper for Logistic Regression to classify audio embeddings.
    """
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def train(self, features, labels):
        """
        Train the logistic regression model.
        
        Args:
            features (np.ndarray): Array of shape (n_samples, 128).
            labels (np.ndarray): Array of shape (n_samples,).
        """
        print("Training Audio Scam Detector...")
        self.classifier.fit(features, labels)
        self.is_trained = True
        print("Training complete.")

    def predict(self, features):
        """
        Predict probability of scam.
        
        Args:
            features (np.ndarray): Array of shape (n_samples, 128) or (128,).
            
        Returns:
            float: Probability of being Real (class 1). 
                   (Or class 0 depending on mapping, usually predict_proba returns [prob_0, prob_1])
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
            
        # Ensure 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Return probability of class 1 (Real)
        probs = self.classifier.predict_proba(features)
        return probs[:, 1] # Probability of being 'Real'

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
            
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
        self.is_trained = True

class AdvancedAudioAnalyzer:
    """
    Advanced audio analysis for detecting voice authenticity and emotional patterns.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.feature_extractor = AudioFeatureExtractor(device)
        self.voice_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.emotion_classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False
        
    def analyze_voice_authenticity(self, audio_path):
        """
        Analyze voice for authenticity markers.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Voice authenticity analysis results
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract comprehensive voice features
            voice_features = self._extract_voice_features(y, sr)
            
            # Analyze temporal consistency
            temporal_features = self.feature_extractor.extract_temporal_features(y, sr)
            
            # Combine all features
            authenticity_score = self._calculate_authenticity_score(
                voice_features, temporal_features
            )
            
            return {
                'authenticity_score': authenticity_score,
                'voice_consistency': temporal_features['temporal_consistency'],
                'pitch_stability': temporal_features['pitch_stability'],
                'spectral_features': voice_features,
                'is_likely_synthetic': authenticity_score < 0.3
            }
            
        except Exception as e:
            print(f"Error analyzing voice authenticity: {e}")
            return {
                'authenticity_score': 0.5,
                'voice_consistency': 0.5,
                'pitch_stability': 0.5,
                'is_likely_synthetic': False
            }
    
    def _extract_voice_features(self, y, sr):
        """
        Extract comprehensive voice characteristics.
        """
        features = {}
        
        # Fundamental frequency (pitch) analysis
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_clean = f0[f0 > 0]  # Remove unvoiced segments
        
        if len(f0_clean) > 0:
            features['pitch_mean'] = np.mean(f0_clean)
            features['pitch_std'] = np.std(f0_clean)
            features['pitch_range'] = np.max(f0_clean) - np.min(f0_clean)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # Voice quality indicators
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Jitter and shimmer (voice stability indicators)
        features['jitter'] = self._calculate_jitter(f0_clean) if len(f0_clean) > 1 else 0
        features['shimmer'] = self._calculate_shimmer(y, sr)
        
        return features
    
    def _calculate_jitter(self, f0):
        """
        Calculate jitter (pitch perturbation).
        """
        if len(f0) < 2:
            return 0
        
        periods = 1 / f0
        period_diffs = np.abs(np.diff(periods))
        mean_period = np.mean(periods[:-1])
        
        jitter = np.mean(period_diffs) / mean_period if mean_period > 0 else 0
        return min(jitter, 1.0)  # Cap at 1.0 for stability
    
    def _calculate_shimmer(self, y, sr, frame_length=1024):
        """
        Calculate shimmer (amplitude perturbation).
        """
        # Calculate RMS energy for each frame
        hop_length = frame_length // 4
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        if len(rms) < 2:
            return 0
        
        # Calculate relative amplitude differences
        rms_diffs = np.abs(np.diff(rms))
        mean_rms = np.mean(rms[:-1])
        
        shimmer = np.mean(rms_diffs) / mean_rms if mean_rms > 0 else 0
        return min(shimmer, 1.0)  # Cap at 1.0 for stability
    
    def _calculate_authenticity_score(self, voice_features, temporal_features):
        """
        Calculate overall voice authenticity score.
        """
        # Weight different factors
        consistency_weight = 0.3
        pitch_stability_weight = 0.25
        spectral_weight = 0.25
        quality_weight = 0.2
        
        # Consistency score
        consistency_score = temporal_features['temporal_consistency']
        
        # Pitch stability score
        pitch_stability = temporal_features['pitch_stability']
        
        # Spectral naturalness (lower jitter/shimmer = more natural)
        jitter_score = max(0, 1.0 - voice_features['jitter'] * 10)  # Scale jitter
        shimmer_score = max(0, 1.0 - voice_features['shimmer'] * 10)  # Scale shimmer
        spectral_score = (jitter_score + shimmer_score) / 2
        
        # Voice quality score based on pitch range and spectral features
        pitch_range_norm = min(1.0, voice_features['pitch_range'] / 200.0)  # Normalize to reasonable range
        quality_score = pitch_range_norm
        
        # Combine all scores
        authenticity_score = (
            consistency_weight * consistency_score +
            pitch_stability_weight * pitch_stability +
            spectral_weight * spectral_score +
            quality_weight * quality_score
        )
        
        return max(0, min(1, authenticity_score))
    
    def analyze_emotional_patterns(self, audio_path):
        """
        Analyze emotional authenticity in speech.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Emotional pattern analysis
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract emotional indicators
            emotional_features = {
                'energy_dynamics': self._analyze_energy_dynamics(y),
                'pitch_dynamics': self._analyze_pitch_dynamics(y, sr),
                'speech_rate': self._estimate_speech_rate(y, sr),
                'pause_patterns': self._analyze_pause_patterns(y, sr)
            }
            
            # Calculate emotional authenticity score
            authenticity = self._calculate_emotional_authenticity(emotional_features)
            
            return {
                'emotional_authenticity': authenticity,
                'energy_variation': emotional_features['energy_dynamics'],
                'pitch_variation': emotional_features['pitch_dynamics'],
                'speech_naturalness': emotional_features['speech_rate'],
                'is_likely_scripted': authenticity < 0.4
            }
            
        except Exception as e:
            print(f"Error analyzing emotional patterns: {e}")
            return {
                'emotional_authenticity': 0.5,
                'energy_variation': 0.5,
                'pitch_variation': 0.5,
                'speech_naturalness': 0.5,
                'is_likely_scripted': False
            }
    
    def _analyze_energy_dynamics(self, y, frame_length=2048, hop_length=512):
        """
        Analyze energy variation patterns.
        """
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate energy variation metrics
        energy_range = np.max(rms) - np.min(rms)
        energy_std = np.std(rms)
        energy_mean = np.mean(rms)
        
        # Normalize energy variation (higher variation = more natural)
        variation_score = min(1.0, energy_std / (energy_mean + 1e-6))
        
        return variation_score
    
    def _analyze_pitch_dynamics(self, y, sr):
        """
        Analyze pitch variation patterns.
        """
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_clean = f0[f0 > 0]
        
        if len(f0_clean) < 10:
            return 0.5
        
        # Calculate pitch variation metrics
        pitch_range = np.max(f0_clean) - np.min(f0_clean)
        pitch_std = np.std(f0_clean)
        pitch_mean = np.mean(f0_clean)
        
        # Natural speech should have reasonable pitch variation
        variation_score = min(1.0, pitch_std / (pitch_mean * 0.1 + 1e-6))
        
        return variation_score
    
    def _estimate_speech_rate(self, y, sr, frame_length=2048, hop_length=512):
        """
        Estimate speech rate and naturalness.
        """
        # Simple speech rate estimation using energy-based voice activity detection
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold for voice activity
        threshold = np.mean(rms) * 0.1
        voiced_frames = rms > threshold
        
        # Estimate speech segments
        speech_ratio = np.sum(voiced_frames) / len(voiced_frames)
        
        # Natural speech typically has 40-70% voice activity
        if 0.4 <= speech_ratio <= 0.7:
            naturalness = 1.0
        else:
            naturalness = max(0, 1.0 - abs(speech_ratio - 0.55) * 2)
        
        return naturalness
    
    def _analyze_pause_patterns(self, y, sr, frame_length=2048, hop_length=512):
        """
        Analyze pause patterns in speech.
        """
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Detect pauses (low energy regions)
        threshold = np.mean(rms) * 0.1
        silence_frames = rms <= threshold
        
        # Find pause segments
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                pause_length = (i - pause_start) * hop_length / sr  # Convert to seconds
                pauses.append(pause_length)
        
        if len(pauses) == 0:
            return 0.3  # No pauses might indicate synthetic speech
        
        # Analyze pause distribution
        pause_mean = np.mean(pauses)
        pause_std = np.std(pauses)
        
        # Natural speech has varied but reasonable pause lengths
        if 0.1 <= pause_mean <= 2.0 and pause_std > 0.05:
            return min(1.0, pause_std / pause_mean)
        else:
            return 0.5
    
    def _calculate_emotional_authenticity(self, features):
        """
        Calculate overall emotional authenticity score.
        """
        scores = [
            features['energy_dynamics'],
            features['pitch_dynamics'],
            features['speech_rate'],
            features['pause_patterns']
        ]
        
        return np.mean(scores)
    
    def get_comprehensive_audio_features(self, audio_path):
        """
        Extract comprehensive audio features for fusion analysis.
        
        Returns:
            np.ndarray: Combined feature vector for fusion
        """
        # VGGish embeddings
        vggish_features = self.feature_extractor.process_audio(audio_path)
        
        # Voice authenticity features
        voice_analysis = self.analyze_voice_authenticity(audio_path)
        
        # Emotional pattern features
        emotion_analysis = self.analyze_emotional_patterns(audio_path)
        
        # Combine into single feature vector
        combined_features = np.concatenate([
            vggish_features,  # 128 dimensions
            [voice_analysis['authenticity_score']],
            [voice_analysis['voice_consistency']],
            [voice_analysis['pitch_stability']],
            [emotion_analysis['emotional_authenticity']],
            [emotion_analysis['energy_variation']],
            [emotion_analysis['pitch_variation']],
            [emotion_analysis['speech_naturalness']]
        ])  # Total: 135 dimensions
        
        return combined_features
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
        self.is_trained = True
