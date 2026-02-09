import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from multimodal_fusion import MultimodalScamDetector

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
REAL_PATH = os.path.join(DATASET_PATH, "real_videos")
SCAM_PATH = os.path.join(DATASET_PATH, "scam_videos")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "weights")
BATCH_SIZE = 2  # Smaller batch size for memory efficiency
EPOCHS = 25
LEARNING_RATE = 0.0001

class EnhancedVideoDataset(Dataset):
    """
    Enhanced dataset that works with the multimodal fusion system.
    """
    def __init__(self, real_dir, scam_dir):
        self.samples = []
        
        # 1. Load Real Videos (Label = 1.0)
        real_videos = glob.glob(os.path.join(real_dir, "*.*"))
        for video in real_videos:
            if video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                self.samples.append((video, 1.0))
            
        # 2. Load Scam Videos (Label = 0.0)
        scam_videos = glob.glob(os.path.join(scam_dir, "*.*"))
        for video in scam_videos:
            if video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                self.samples.append((video, 0.0))
            
        # Shuffle data
        random.shuffle(self.samples)
        
        print(f"Found {len(real_videos)} Real videos and {len(scam_videos)} Scam videos.")
        print(f"Total valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        return video_path, label

def train_enhanced_system():
    """
    Train the enhanced multimodal system with all components.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Initialize Enhanced System
    print("Initializing Enhanced Multimodal System...")
    detector = MultimodalScamDetector(device=device)
    
    # 3. Setup Data
    dataset = EnhancedVideoDataset(REAL_PATH, SCAM_PATH)
    if len(dataset) == 0:
        print("Error: No videos found. Please add videos to dataset/real_videos and dataset/scam_videos")
        return

    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # 4. Training Phase 1: Visual Quality Head
    print("\n" + "="*60)
    print("PHASE 1: Training Visual Quality Head")
    print("="*60)
    
    visual_criterion = nn.BCELoss()
    visual_optimizer = optim.Adam(detector.visual_analyzer.head.parameters(), lr=LEARNING_RATE * 5)
    
    detector.visual_analyzer.train()
    for epoch in range(10):  # Pre-train visual head
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, len(train_dataset), BATCH_SIZE):
            batch_videos = []
            batch_labels = []
            
            for j in range(i, min(i + BATCH_SIZE, len(train_dataset))):
                video_path, label = train_dataset[j]
                batch_videos.append(video_path)
                batch_labels.append(label)
            
            try:
                # Process batch
                batch_features = []
                for video_path in batch_videos:
                    visual_features = detector.extract_visual_features(video_path)
                    # Use just the quality score for this phase
                    quality_score = visual_features[128] if len(visual_features) > 128 else 0.5
                    batch_features.append(quality_score)
                
                predictions = torch.FloatTensor(batch_features).unsqueeze(1).to(device)
                labels = torch.FloatTensor(batch_labels).unsqueeze(1).to(device)
                
                loss = visual_criterion(predictions, labels)
                
                visual_optimizer.zero_grad()
                loss.backward()
                visual_optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            print(f"Visual Epoch {epoch + 1}/10 - Loss: {avg_loss:.4f}")
    
    # 5. Training Phase 2: Multimodal Fusion
    print("\n" + "="*60)
    print("PHASE 2: Training Multimodal Fusion Network")
    print("="*60)
    
    # Prepare data for fusion training
    train_videos = []
    train_labels = []
    
    print("Extracting features for fusion training...")
    for i in range(len(train_dataset)):
        video_path, label = train_dataset[i]
        train_videos.append(video_path)
        train_labels.append(label)
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(train_dataset)} videos...")
    
    # Train fusion network
    try:
        detector.train_fusion_network(
            video_paths=train_videos,
            labels=train_labels,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE
        )
    except Exception as e:
        print(f"Error training fusion network: {e}")
        print("Continuing with pre-trained components...")
    
    # 6. Validation
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    detector.fusion_network.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i in range(min(10, len(val_dataset))):  # Test on first 10 validation samples
            video_path, true_label = val_dataset[i]
            
            try:
                result = detector.analyze_video(video_path)
                predicted_score = result['overall_score']
                predicted_label = 1 if predicted_score > 0.5 else 0
                
                if predicted_label == true_label:
                    correct_predictions += 1
                
                total_predictions += 1
                
                print(f"Video: {os.path.basename(video_path)}")
                print(f"True: {int(true_label)} | Predicted: {predicted_label} | Score: {predicted_score:.4f}")
                print(f"Result: {'✓' if predicted_label == true_label else '✗'}")
                print("-" * 40)
                
            except Exception as e:
                print(f"Error validating {video_path}: {e}")
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"Validation Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    # 7. Save Models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    detector.save_models(MODEL_SAVE_PATH)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Models saved to: {MODEL_SAVE_PATH}")
    print("\nYou can now use the enhanced system with:")
    print("python test_enhanced_system.py <path_to_video>")

def train_individual_components():
    """
    Train individual components separately for better control.
    """
    print("="*60)
    print("INDIVIDUAL COMPONENT TRAINING")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Train Visual Components
    print("\n1. Training Visual Quality Head...")
    from visual_engine import VisualQualityHead, extract_quality_frames
    
    visual_model = VisualQualityHead().to(device)
    visual_criterion = nn.BCELoss()
    visual_optimizer = optim.Adam(visual_model.head.parameters(), lr=0.001)
    
    dataset = EnhancedVideoDataset(REAL_PATH, SCAM_PATH)
    if len(dataset) == 0:
        print("No training data found!")
        return
    
    visual_model.train()
    for epoch in range(5):
        epoch_loss = 0
        processed = 0
        
        for video_path, label in dataset:
            try:
                frames = extract_quality_frames(video_path, num_frames=5)
                frames = frames.to(device)
                
                with torch.no_grad():
                    scores = visual_model(frames)
                    avg_score = torch.mean(scores)
                
                # Train on average score
                prediction = avg_score.unsqueeze(0)
                target = torch.FloatTensor([label]).to(device)
                
                loss = visual_criterion(prediction, target)
                
                visual_optimizer.zero_grad()
                loss.backward()
                visual_optimizer.step()
                
                epoch_loss += loss.item()
                processed += 1
                
                if processed >= 20:  # Limit for demo
                    break
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        if processed > 0:
            print(f"Epoch {epoch + 1}/5 - Loss: {epoch_loss / processed:.4f}")
    
    # Save visual model
    visual_save_path = os.path.join(MODEL_SAVE_PATH, "visual_quality_head.pth")
    os.makedirs(os.path.dirname(visual_save_path), exist_ok=True)
    visual_model.save_head_weights(visual_save_path)
    
    print("Visual training complete!")

if __name__ == "__main__":
    print("Enhanced Multimodal Training System")
    print("Choose training mode:")
    print("1. Full Enhanced System (recommended)")
    print("2. Individual Components Only")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        train_individual_components()
    else:
        train_enhanced_system()