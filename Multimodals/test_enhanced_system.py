import sys
import os
import torch
from multimodal_fusion import MultimodalScamDetector, analyze_video_quick

def test_enhanced_system(video_path):
    """
    Test the enhanced multimodal scam detection system.
    
    Args:
        video_path (str): Path to video file
    """
    print("="*80)
    print("ENHANCED MULTIMODAL SCAM DETECTION SYSTEM")
    print("="*80)
    print(f"Analyzing: {video_path}")
    print("-"*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize enhanced detector
    detector = MultimodalScamDetector(device=device)
    
    # Load models if available
    model_path = os.path.join("models", "weights")
    if os.path.exists(model_path):
        detector.load_models(model_path)
        print("✓ Loaded trained models")
    else:
        print("⚠ Using untrained models (for demonstration)")
    
    try:
        # Comprehensive analysis
        results = detector.analyze_video(video_path)
        
        # Display results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        
        print(f"Overall Score: {results['overall_score']:.4f}")
        print(f"Confidence: {results['confidence']:.4f}")
        print(f"Verdict: {results['verdict']}")
        print(f"Risk Level: {results['risk_level']}")
        
        print("\n" + "-"*30)
        print("MODALITY BREAKDOWN")
        print("-"*30)
        print(f"Visual Quality: {results['visual_score']:.4f}")
        print(f"Audio Authenticity: {results['audio_score']:.4f}")
        
        print("\n" + "-"*30)
        print("VOICE ANALYSIS")
        print("-"*30)
        voice = results['voice_analysis']
        print(f"Authenticity Score: {voice['authenticity_score']:.4f}")
        print(f"Voice Consistency: {voice['voice_consistency']:.4f}")
        print(f"Pitch Stability: {voice['pitch_stability']:.4f}")
        print(f"Likely Synthetic: {voice['is_likely_synthetic']}")
        
        print("\n" + "-"*30)
        print("EMOTIONAL ANALYSIS")
        print("-"*30)
        emotion = results['emotion_analysis']
        print(f"Emotional Authenticity: {emotion['emotional_authenticity']:.4f}")
        print(f"Energy Variation: {emotion['energy_variation']:.4f}")
        print(f"Pitch Variation: {emotion['pitch_variation']:.4f}")
        print(f"Speech Naturalness: {emotion['speech_naturalness']:.4f}")
        print(f"Likely Scripted: {emotion['is_likely_scripted']}")
        
        print("\n" + "-"*30)
        print("DETAILED SCORES")
        print("-"*30)
        detailed = results['detailed_scores']
        for key, value in detailed.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        # Risk assessment
        print("\n" + "="*50)
        print("RISK ASSESSMENT")
        print("="*50)
        
        if results['risk_level'] == 'HIGH':
            print("🚨 HIGH RISK: Strong indicators of scam/low effort content")
            print("   Recommendations:")
            print("   - Exercise extreme caution")
            print("   - Verify claims through independent sources")
            print("   - Look for additional red flags")
        elif results['risk_level'] == 'MEDIUM':
            print("⚠️  MEDIUM RISK: Mixed indicators detected")
            print("   Recommendations:")
            print("   - Proceed with caution")
            print("   - Cross-check information")
            print("   - Consider additional verification")
        else:
            print("✅ LOW RISK: Appears to be legitimate content")
            print("   Recommendations:")
            print("   - Content appears authentic")
            print("   - Standard verification still recommended")
        
        # Technical details
        print("\n" + "-"*30)
        print("TECHNICAL ANALYSIS")
        print("-"*30)
        
        # Voice technical details
        if 'spectral_features' in voice:
            spectral = voice['spectral_features']
            if isinstance(spectral, dict) and 'jitter' in spectral and 'shimmer' in spectral:
                print(f"Voice Jitter: {spectral['jitter']:.6f}")
                print(f"Voice Shimmer: {spectral['shimmer']:.6f}")
        
        print(f"Voice Consistency: {voice['voice_consistency']:.4f}")
        print(f"Pitch Stability: {voice['pitch_stability']:.4f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("This might be due to:")
        print("- Unsupported video format")
        print("- Missing audio track")
        print("- Corrupted video file")
        print("- Insufficient system resources")

def test_individual_components(video_path):
    """
    Test individual components separately for debugging.
    
    Args:
        video_path (str): Path to video file
    """
    print("\n" + "="*80)
    print("INDIVIDUAL COMPONENT TESTING")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Test visual analysis
        print("\n1. Testing Visual Analysis...")
        from visual_engine import VisualQualityHead, extract_quality_frames
        
        visual_analyzer = VisualQualityHead().to(device)
        frames = extract_quality_frames(video_path, num_frames=5)
        frames = frames.to(device)
        
        with torch.no_grad():
            quality_scores = visual_analyzer(frames)
            print(f"   Visual Quality Scores: {quality_scores.cpu().numpy()}")
            print(f"   Average Quality: {torch.mean(quality_scores).item():.4f}")
        
        # Test lighting analysis
        lighting = visual_analyzer.analyze_lighting_consistency(frames)
        print(f"   Lighting Consistency: {lighting}")
        
        # Test background analysis
        background = visual_analyzer.analyze_background_consistency(frames)
        print(f"   Background Analysis: {background}")
        
    except Exception as e:
        print(f"   Visual analysis failed: {e}")
    
    try:
        # Test audio analysis
        print("\n2. Testing Audio Analysis...")
        from audio_engine import AdvancedAudioAnalyzer
        
        audio_analyzer = AdvancedAudioAnalyzer(device)
        
        # Voice authenticity
        voice_result = audio_analyzer.analyze_voice_authenticity(video_path)
        print(f"   Voice Authenticity: {voice_result['authenticity_score']:.4f}")
        print(f"   Voice Consistency: {voice_result['voice_consistency']:.4f}")
        
        # Emotional patterns
        emotion_result = audio_analyzer.analyze_emotional_patterns(video_path)
        print(f"   Emotional Authenticity: {emotion_result['emotional_authenticity']:.4f}")
        print(f"   Speech Naturalness: {emotion_result['speech_naturalness']:.4f}")
        
    except Exception as e:
        print(f"   Audio analysis failed: {e}")

def analyze_sample_videos():
    """
    Analyze sample videos if they exist in the dataset.
    """
    print("\n" + "="*80)
    print("ANALYZING SAMPLE VIDEOS")
    print("="*80)
    
    # Check for sample videos
    real_dir = os.path.join("dataset", "real_videos")
    scam_dir = os.path.join("dataset", "scam_videos")
    
    sample_videos = []
    
    if os.path.exists(real_dir):
        real_videos = [f for f in os.listdir(real_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        sample_videos.extend([(os.path.join(real_dir, f), "REAL") for f in real_videos[:2]])
    
    if os.path.exists(scam_dir):
        scam_videos = [f for f in os.listdir(scam_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        sample_videos.extend([(os.path.join(scam_dir, f), "SCAM") for f in scam_videos[:2]])
    
    if sample_videos:
        for video_path, expected_label in sample_videos:
            print(f"\nAnalyzing {expected_label} video: {os.path.basename(video_path)}")
            print("-" * 60)
            
            # Quick analysis
            try:
                result = analyze_video_quick(video_path)
                print(f"Score: {result['overall_score']:.4f}")
                print(f"Verdict: {result['verdict']}")
                print(f"Expected: {expected_label}")
                
                # Check if prediction matches expectation
                is_correct = (
                    (expected_label == "REAL" and result['overall_score'] > 0.5) or
                    (expected_label == "SCAM" and result['overall_score'] <= 0.5)
                )
                print(f"Prediction: {'✓ Correct' if is_correct else '✗ Incorrect'}")
                
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("No sample videos found in dataset directories.")
        print("Please add videos to:")
        print("- dataset/real_videos/")
        print("- dataset/scam_videos/")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze specific video
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            test_enhanced_system(video_path)
            test_individual_components(video_path)
        else:
            print(f"Error: Video file not found: {video_path}")
    else:
        # Run sample analysis
        print("Enhanced Multimodal Scam Detection System")
        print("Usage: python test_enhanced_system.py <path_to_video>")
        print("\nRunning analysis on sample videos...")
        analyze_sample_videos()