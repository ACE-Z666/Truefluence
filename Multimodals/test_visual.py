import torch
import sys
import os
from visual_engine import VisualQualityHead, extract_quality_frames

# Configuration
WEIGHTS_PATH = os.path.join("models", "weights", "visual_quality_head.pth")

def test_video(video_path):
    print(f"\n--- Testing Video: {video_path} ---")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    print("Loading model...")
    model = VisualQualityHead().to(device)
    
    # Load the trained weights
    # We need to map location to cpu if you trained on gpu but test on cpu
    try:
        if os.path.exists(WEIGHTS_PATH):
            model.load_head_weights(WEIGHTS_PATH)
            model.eval() # Set to evaluation mode (turns off Dropout)
        else:
            print(f"Error: Weights file not found at {WEIGHTS_PATH}")
            print("Please run train_visual.py first.")
            return
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 3. Process Video
    try:
        print("Extracting frames...")
        # Get tensor of shape (10, 3, 224, 224)
        frames = extract_quality_frames(video_path, num_frames=10)
        frames = frames.to(device)
        
        # 4. Predict
        print("Analyzing visual quality...")
        with torch.no_grad():
            # Get scores for all 10 frames
            scores = model(frames)
            
            # Calculate average score for the whole video
            avg_score = torch.mean(scores).item()
            
        # 5. Output Results
        print("\n" + "="*30)
        print(f"VISUAL QUALITY SCORE: {avg_score:.4f}")
        print("="*30)
        
        if avg_score > 0.5:
             print("✅ VERDICT: REAL / PROFESSIONAL")
             print(f"Confidence: {avg_score * 100:.1f}%")
        else:
             print("❌ VERDICT: SCAM / LOW EFFORT")
             print(f"Confidence: {(1 - avg_score) * 100:.1f}%")
             
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_visual.py <path_to_video>")
    else:
        video_path = sys.argv[1]
        test_video(video_path)
