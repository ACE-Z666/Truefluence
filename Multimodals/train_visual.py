import os
import torch
import torch.nn as nn
import torch.optim as optim
from visual_engine import VisualQualityHead, extract_quality_frames, analyze_video_quick
from audio_engine import analyze_audio_complete
import glob
import random
from tqdm import tqdm
import time

def train_multimodal_system():
    """Multimodal training with temporal and fusion features."""
    print("="*70)
    print("MULTIMODAL SCAM DETECTION - TRAINING SYSTEM")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load enhanced model
    print("Loading enhanced model...")
    model = VisualQualityHead().to(device)
    
    # Check for pre-trained enhanced weights
    enhanced_weights = os.path.join("models", "weights", "enhanced_visual_model.pth")
    if os.path.exists(enhanced_weights):
        print("Loading pre-trained enhanced weights...")
        model.load_head_weights(enhanced_weights)
        print("✓ Pre-trained enhanced weights loaded")
    
    print("✓ Enhanced model ready")
    
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
    
    # Enhanced training setup
    criterion = nn.BCELoss()
    
    # Separate optimizers for different components
    basic_optimizer = optim.Adam(model.head.parameters(), lr=0.001)
    temporal_optimizer = optim.Adam(list(model.temporal_lstm.parameters()) + 
                                   list(model.temporal_attention.parameters()), lr=0.0005)
    fusion_optimizer = optim.Adam(model.fusion_network.parameters(), lr=0.0005)
    
    epochs = 5
    print(f"\nStarting enhanced training for {epochs} epochs...")
    print("-" * 70)
    
    model.train()
    
    for epoch in range(epochs):
        print(f"\n🔥 EPOCH {epoch + 1}/{epochs} - ENHANCED MODE")
        print("-" * 50)
        
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        temporal_losses = []
        fusion_losses = []
        
        for i, (video_path, label) in enumerate(tqdm(zip(videos, labels), 
                                                    total=len(videos), 
                                                    desc=f"Enhanced Epoch {epoch+1}")):
            try:
                print(f"Processing: {os.path.basename(video_path)} (Label: {label})")
                
                # 1. Basic visual training
                start_time = time.time()
                frames = extract_quality_frames(video_path, num_frames=8)  # More frames for temporal analysis
                frames = frames.to(device)
                
                basic_optimizer.zero_grad()
                
                # Basic scores
                scores = model(frames)
                avg_score = torch.mean(scores)
                
                target = torch.tensor([label], dtype=torch.float32).to(device)
                basic_loss = criterion(avg_score.unsqueeze(0), target)
                basic_loss.backward(retain_graph=True)
                basic_optimizer.step()
                
                # 2. Temporal analysis training
                temporal_optimizer.zero_grad()
                
                batch_frames = frames.unsqueeze(0)  # Add batch dimension
                temporal_output = model.forward_temporal(batch_frames)
                temporal_loss = criterion(temporal_output, target.unsqueeze(0))
                
                temporal_loss.backward(retain_graph=True)
                temporal_optimizer.step()
                temporal_losses.append(temporal_loss.item())
                
                # 3. Multimodal fusion training (if audio available)
                try:
                    audio_results = analyze_audio_complete(video_path, device=str(device))
                    audio_features = audio_results['audio_features']
                    
                    if audio_features is not None:
                        fusion_optimizer.zero_grad()
                        
                        fusion_output = model.forward_fusion(batch_frames, audio_features.unsqueeze(0).to(device))
                        fusion_loss = criterion(fusion_output, target.unsqueeze(0))
                        
                        fusion_loss.backward()
                        fusion_optimizer.step()
                        fusion_losses.append(fusion_loss.item())
                    else:
                        fusion_losses.append(0.0)
                        
                except Exception as audio_error:
                    print(f"    ⚠ Audio processing failed: {audio_error}")
                    fusion_losses.append(0.0)
                
                process_time = time.time() - start_time
                
                # Track metrics with enhanced components
                total_loss = basic_loss.item() + temporal_loss.item()
                if fusion_losses[-1] > 0:
                    total_loss += fusion_losses[-1]
                
                epoch_loss += total_loss
                predicted = (avg_score > 0.5).float()
                correct = (predicted == label).float()
                correct_predictions += correct.item()
                total_predictions += 1
                
                # Enhanced feedback
                print(f"  Basic Score: {avg_score.item():.4f}")
                print(f"  Basic Loss: {basic_loss.item():.4f}")
                print(f"  Temporal Loss: {temporal_loss.item():.4f}")
                if fusion_losses[-1] > 0:
                    print(f"  Fusion Loss: {fusion_losses[-1]:.4f}")
                print(f"  Correct: {correct.item()}, Time: {process_time:.2f}s")
                
                # Progress reports
                if (i + 1) % 3 == 0:
                    avg_loss = epoch_loss / (i + 1)
                    accuracy = correct_predictions / total_predictions
                    avg_temporal = sum(temporal_losses) / len(temporal_losses)
                    avg_fusion = sum(fusion_losses) / len(fusion_losses) if fusion_losses else 0
                    
                    print(f"  📊 Progress: {i+1}/{len(videos)} videos")
                    print(f"     Avg Total Loss: {avg_loss:.4f}")
                    print(f"     Accuracy: {accuracy:.4f}")
                    print(f"     Avg Temporal Loss: {avg_temporal:.4f}")
                    if avg_fusion > 0:
                        print(f"     Avg Fusion Loss: {avg_fusion:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error with {video_path}: {e}")
                continue
        
        # Enhanced epoch summary
        if total_predictions > 0:
            avg_epoch_loss = epoch_loss / total_predictions
            epoch_accuracy = correct_predictions / total_predictions
            avg_temporal = sum(temporal_losses) / len(temporal_losses) if temporal_losses else 0
            avg_fusion = sum(fusion_losses) / len(fusion_losses) if fusion_losses else 0
            
            print(f"\n✅ Enhanced Epoch {epoch + 1} Complete:")
            print(f"   Average Total Loss: {avg_epoch_loss:.4f}")
            print(f"   Accuracy: {epoch_accuracy:.4f}")
            print(f"   Average Temporal Loss: {avg_temporal:.4f}")
            if avg_fusion > 0:
                print(f"   Average Fusion Loss: {avg_fusion:.4f}")
                print(f"   Multimodal Training: Active")
            else:
                print(f"   Multimodal Training: Audio unavailable")
            print(f"   Processed: {total_predictions}/{len(videos)} videos")
        
        # Save model
        model_path = os.path.join("models", "weights", f"enhanced_visual_epoch_{epoch+1}.pth")
        final_model_path = os.path.join("models", "weights", "enhanced_visual_model.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_head_weights(model_path)
        model.save_head_weights(final_model_path)  # Always update the main model
        print(f"   💾 Model saved: {model_path}")
        print(f"   💾 Main model updated: {final_model_path}")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("Use 'python test_visual.py <video_path>' to test the enhanced system")
    print("="*70)



if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("🚀 MULTIMODAL SCAM DETECTION TRAINING SYSTEM")
    print("Enhanced with Temporal Analysis & Audio Fusion")
    print("="*70)
    print()
    
    train_multimodal_system()
    simple_training()