import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from visual_engine import VisualQualityHead, extract_quality_frames
import random
import glob

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
REAL_PATH = os.path.join(DATASET_PATH, "real_videos")
SCAM_PATH = os.path.join(DATASET_PATH, "scam_videos")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "weights", "visual_quality_head.pth")
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

class VideoDataset(Dataset):
    def __init__(self, real_dir, scam_dir):
        self.samples = []
        
        # 1. Load Real Videos (Label = 1.0)
        real_videos = glob.glob(os.path.join(real_dir, "*.*"))
        for video in real_videos:
            self.samples.append((video, 1.0))
            
        # 2. Load Scam Videos (Label = 0.0)
        scam_videos = glob.glob(os.path.join(scam_dir, "*.*"))
        for video in scam_videos:
            self.samples.append((video, 0.0))
            
        # Shuffle data
        random.shuffle(self.samples)
        
        print(f"Found {len(real_videos)} Real videos and {len(scam_videos)} Scam videos.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            # Extract frames using the engine's helper function
            # This returns tensor of shape (Batch, 3, 224, 224)
            frames_tensor = extract_quality_frames(video_path, num_frames=10)
            return frames_tensor, torch.tensor([label], dtype=torch.float32)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return a dummy tensor or handle appropriately (simple skip logic here usually requires custom collate)
            # For simplicity, we retun zero tensors (the loop will need to be robust or we ensure data is clean)
            return torch.zeros((10, 3, 224, 224)), torch.tensor([label], dtype=torch.float32)

def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Initialize Model
    model = VisualQualityHead().to(device)
    
    # 3. Setup Data
    dataset = VideoDataset(REAL_PATH, SCAM_PATH)
    if len(dataset) == 0:
        print("Error: No videos found. Please add videos to dataset/real_videos and dataset/scam_videos")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Loss and Optimizer
    # Since we added Sigmoid() in the model, we use BCELoss (Binary Cross Entropy)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE) # Only train the head!

    # 5. Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)
            
            # Note: extract_quality_frames returns (num_frames, 3, 224, 224) per video.
            # But DataLoader makes it (Batch, num_frames, 3, 224, 224).
            # We need to process each video in the batch.
            
            # Currently VisualQualityHead takes (Batch_Images, Channels, H, W).
            # We treat every frame of the video as a separate datapoint OR average the scores.
            
            # APPROACH: Instance Learning. Pass all 10 frames through, get 10 scores, average them -> Final Video Score.
            
            optimizer.zero_grad()
            
            batch_loss = 0
            for i in range(len(frames)): # Iterate over batch items
                video_frames = frames[i] # Shape: (10, 3, 224, 224)
                video_label = labels[i]  # Shape: (1)
                
                # Forward pass for all frames of this video
                scores = model(video_frames) # Shape: (10, 1)
                
                # Average the scores for the video
                avg_score = torch.mean(scores, dim=0) # Shape: (1)
                
                # Calculate loss for this video
                loss = criterion(avg_score, video_label)
                loss.backward()
                batch_loss += loss.item()
            
            optimizer.step()
            total_loss += batch_loss / len(frames)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f}")

    # 6. Save Weights
    model.save_head_weights(MODEL_SAVE_PATH)
    print("Training complete!")

if __name__ == "__main__":
    train()
