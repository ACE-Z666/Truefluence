import torch
import numpy as np
import librosa
import os
from sklearn.linear_model import LogisticRegression
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
