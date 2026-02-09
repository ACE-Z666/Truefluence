import torch
import sys
import os
from visual_engine import VisualQualityHead, extract_quality_frames, analyze_video_quick
from audio_engine import analyze_audio_complete

# Configuration
ENHANCED_WEIGHTS_PATH = os.path.join("models", "weights", "enhanced_visual_model.pth")

def test_video(video_path):
    print(f"\n--- Enhanced Multimodal Video Analysis: {video_path} ---")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Enhanced Model
    print("Loading enhanced model...")
    model = VisualQualityHead().to(device)
    
    # Load the trained weights
    try:
        if os.path.exists(ENHANCED_WEIGHTS_PATH):
            model.load_head_weights(ENHANCED_WEIGHTS_PATH)
            model.eval()
            print("✓ Enhanced model loaded")
        else:
            print(f"⚠ Enhanced weights not found at {ENHANCED_WEIGHTS_PATH}")
            print("Run 'python train_visual.py' to train the model first.")
            print("Using untrained model for demonstration.")
            model.eval()
    except Exception as e:
        print(f"Error loading weights: {e}")
        model.eval()

    # 3. Enhanced Analysis
    try:
        print("\n🔍 COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        # Basic visual analysis
        print("1. Extracting visual features...")
        frames = extract_quality_frames(video_path, num_frames=10)
        frames = frames.to(device)
        
        with torch.no_grad():
            # Basic quality scores
            scores = model(frames)
            avg_score = torch.mean(scores).item()
            
            # Enhanced visual features
            visual_features = model.extract_comprehensive_visual_features(video_path)
        
        print(f"   ✓ Visual quality score: {avg_score:.4f}")
        
        # Audio analysis
        print("2. Analyzing audio...")
        try:
            audio_results = analyze_audio_complete(video_path, device=str(device))
            audio_features = audio_results['audio_features']
            has_audio = True
            print(f"   ✓ Audio authenticity: {audio_results['overall_audio_score']:.4f}")
        except Exception as e:
            print(f"   ⚠ Audio analysis failed: {e}")
            audio_features = None
            has_audio = False
        
        # Complete multimodal analysis
        print("3. Multimodal fusion...")
        complete_results = model.analyze_video_complete(video_path, audio_features)
        
        # 4. Display Results
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        
        print(f"Overall Score: {complete_results['overall_score']:.4f}")
        print(f"Visual Score: {complete_results['visual_score']:.4f}")
        print(f"Confidence: {complete_results['confidence']:.4f}")
        print(f"Verdict: {complete_results['verdict']}")
        print(f"Risk Level: {complete_results['risk_level']}")
        
        if has_audio and audio_features is not None:
            print(f"\n🎤 AUDIO ANALYSIS:")
            voice_analysis = audio_results['voice_analysis']
            emotion_analysis = audio_results['emotion_analysis']
            
            print(f"Voice Authenticity: {voice_analysis['authenticity_score']:.4f}")
            print(f"Voice Consistency: {voice_analysis['voice_consistency']:.4f}")
            print(f"Emotional Authenticity: {emotion_analysis['emotional_authenticity']:.4f}")
            print(f"Speech Naturalness: {emotion_analysis['speech_naturalness']:.4f}")
            print(f"Likely Synthetic Voice: {audio_results['is_likely_synthetic']}")
            print(f"Likely Scripted Speech: {audio_results['is_likely_scripted']}")
        
        print(f"\n🎬 ENHANCED FEATURES:")
        print(f"Multimodal Analysis: {complete_results['has_multimodal']}")
        print(f"Fusion Model Trained: {complete_results['fusion_trained']}")
        
        # Risk Assessment
        print(f"\n🚨 RISK ASSESSMENT:")
        if complete_results['risk_level'] == 'HIGH':
            print("🚨 HIGH RISK - Strong scam indicators detected")
            print("   Recommendation: Exercise extreme caution")
        elif complete_results['risk_level'] == 'MEDIUM':
            print("⚠️ MEDIUM RISK - Mixed indicators")
            print("   Recommendation: Proceed with caution")
        else:
            print("✅ LOW RISK - Appears legitimate")
            print("   Recommendation: Content seems authentic")
        
        # Legacy compatibility note
        print("\n" + "="*30)
        print(f"VISUAL QUALITY SCORE: {avg_score:.4f}")
        print("="*30)
        
        if avg_score > 0.5:
             print("✅ VERDICT: LEGITIMATE / PROFESSIONAL")
             print(f"Confidence: {avg_score * 100:.1f}%")
        else:
             print("❌ VERDICT: SCAM / LOW EFFORT")
             print(f"Confidence: {(1 - avg_score) * 100:.1f}%")
             
    except Exception as e:
        print(f"Error processing video: {e}")
        print("Make sure the video file exists and is in a supported format (MP4, AVI, MOV).")

def test_enhanced_features(video_path):
    """Test specific enhanced features separately."""
    print(f"\n--- Testing Enhanced Features: {video_path} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisualQualityHead().to(device)
    
    if os.path.exists(ENHANCED_WEIGHTS_PATH):
        model.load_head_weights(ENHANCED_WEIGHTS_PATH)
    
    model.eval()
    
    try:
        # Test frame extraction
        print("1. Testing frame extraction...")
        frames = extract_quality_frames(video_path, num_frames=5)
        print(f"   ✓ Extracted {frames.shape[0]} frames")
        
        # Test lighting analysis
        print("2. Testing lighting analysis...")
        lighting = model.analyze_lighting_consistency(frames.to(device))
        print(f"   ✓ Lighting consistency: {lighting['luminance_consistency']:.4f}")
        
        # Test background analysis
        print("3. Testing background analysis...")
        background = model.analyze_background_consistency(frames.to(device))
        print(f"   ✓ Background consistency: {background['corner_consistency']:.4f}")
        
        # Test temporal analysis
        print("4. Testing temporal analysis...")
        temporal = model.analyze_temporal_consistency(frames.unsqueeze(0).to(device))
        print(f"   ✓ Temporal consistency: {temporal['temporal_consistency']:.4f}")
        
        print("\n✅ All enhanced features working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing enhanced features: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enhanced Multimodal Video Analysis System")
        print("Usage: python test_visual.py <path_to_video> [--enhanced]")
        print("  --enhanced : Test individual enhanced features")
        print("")
        print("Example: python test_visual.py suspicious_video.mp4")
    else:
        video_path = sys.argv[1]
        
        if len(sys.argv) > 2 and sys.argv[2] == "--enhanced":
            test_enhanced_features(video_path)
        else:
            test_video(video_path)
