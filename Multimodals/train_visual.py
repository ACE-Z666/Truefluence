import os
import torch
import torch.nn as nn
import torch.optim as optim
from visual_engine import VisualQualityHead, extract_quality_frames
import glob
import random
from tqdm import tqdm
import time

def simple_training():
    """Simplified training with immediate feedback."""
    print("="*60)
    print("SIMPLE VISUAL TRAINING - WITH PROGRESS")
    print("="*60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    model = VisualQualityHead().to(device)
    print("✓ Model loaded")
    
    # Find videos
    real_dir = os.path.join("dataset", "real_videos")
    scam_dir = os.path.join("dataset", "scam_videos")
    
    videos = []
    labels = []
    
    if os.path.exists(real_dir):
        real_videos = glob.glob(os.path.join(real_dir, "*.mp4")) + \
                      glob.glob(os.path.join(real_dir, "*.avi")) + \
                      glob.glob(os.path.join(real_dir, "*.mov"))
        videos.extend(real_videos)
        labels.extend([1.0] * len(real_videos))
        print(f"Real videos: {len(real_videos)}")
    
    if os.path.exists(scam_dir):
        scam_videos = glob.glob(os.path.join(scam_dir, "*.mp4")) + \
                      glob.glob(os.path.join(scam_dir, "*.avi")) + \
                      glob.glob(os.path.join(scam_dir, "*.mov"))
        videos.extend(scam_videos)
        labels.extend([0.0] * len(scam_videos))
        print(f"Scam videos: {len(scam_videos)}")
    
    if len(videos) == 0:
        print("❌ No videos found!")
        print("Add videos to dataset/real_videos/ and dataset/scam_videos/")
        return
    
    print(f"Total videos: {len(videos)}")
    
    # Shuffle data
    combined = list(zip(videos, labels))
    random.shuffle(combined)
    videos, labels = zip(*combined)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.head.parameters(), lr=0.001)
    
    epochs = 5  # Start small
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    model.train()
    
    for epoch in range(epochs):
        print(f"\n🔥 EPOCH {epoch + 1}/{epochs}")
        print("-" * 40)
        
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Process videos one by one with progress bar
        for i, (video_path, label) in enumerate(tqdm(zip(videos, labels), 
                                                    total=len(videos), 
                                                    desc=f"Epoch {epoch+1}")):
            try:
                # Load and process video
                print(f"Processing: {os.path.basename(video_path)} (Label: {label})")
                
                start_time = time.time()
                frames = extract_quality_frames(video_path, num_frames=5)  # Fewer frames for speed
                frames = frames.to(device)
                load_time = time.time() - start_time
                
                # Forward pass
                optimizer.zero_grad()
                
                scores = model(frames)  # Shape: (5, 1)
                avg_score = torch.mean(scores)  # Average across frames
                
                # Calculate loss
                target = torch.tensor([label], dtype=torch.float32).to(device)
                loss = criterion(avg_score.unsqueeze(0), target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predicted = (avg_score > 0.5).float()
                correct = (predicted == label).float()
                correct_predictions += correct.item()
                total_predictions += 1
                
                # Show immediate feedback
                print(f"  Score: {avg_score.item():.4f}, Loss: {loss.item():.4f}, "
                      f"Correct: {correct.item()}, Load time: {load_time:.2f}s")
                
                # Save progress every 5 videos
                if (i + 1) % 5 == 0:
                    avg_loss = epoch_loss / (i + 1)
                    accuracy = correct_predictions / total_predictions
                    print(f"  📊 Progress: {i+1}/{len(videos)} videos, "
                          f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error with {video_path}: {e}")
                continue
        
        # Epoch summary
        if total_predictions > 0:
            avg_epoch_loss = epoch_loss / total_predictions
            epoch_accuracy = correct_predictions / total_predictions
            print(f"\n✅ Epoch {epoch + 1} Complete:")
            print(f"   Average Loss: {avg_epoch_loss:.4f}")
            print(f"   Accuracy: {epoch_accuracy:.4f}")
            print(f"   Processed: {total_predictions}/{len(videos)} videos")
        
        # Save model after each epoch
        model_path = os.path.join("models", "weights", f"visual_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_head_weights(model_path)
        print(f"   💾 Model saved: {model_path}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    simple_training()