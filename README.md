# Enhanced Multimodal Scam Detection System

A sophisticated AI system that combines visual and audio analysis to detect scam videos with high accuracy using temporal analysis, voice authenticity detection, and advanced fusion techniques.

## 🚀 Key Features

### Enhanced Visual Analysis
- **Temporal Consistency Analysis**: LSTM-based temporal patterns detection
- **Face Region Detection**: Skin color analysis for face presence
- **Lighting Consistency**: Professional vs amateur lighting detection
- **Background Analysis**: Compositing and green screen artifact detection
- **Cross-frame Correlation**: Detect editing artifacts and inconsistencies

### Advanced Audio Analysis
- **Voice Authenticity Detection**: Jitter, shimmer, and pitch stability analysis
- **Emotional Pattern Recognition**: Natural vs scripted speech detection
- **Speech Rate Analysis**: Naturalness assessment
- **Pause Pattern Analysis**: Authentic conversation flow detection
- **VGGish Feature Extraction**: Deep audio embeddings for classification

### Multimodal Fusion Network
- **Cross-Modal Attention**: Learn relationships between visual and audio cues
- **Adaptive Feature Weighting**: Automatically balance modality importance
- **Confidence Scoring**: Uncertainty quantification for reliable predictions
- **Risk Assessment**: Multi-level risk categorization (Low/Medium/High)

## 📁 Project Structure

```
Multimodals/
├── visual_engine.py          # 🔥 ENHANCED: Integrated multimodal system with temporal & fusion
├── audio_engine.py           # 🔥 ENHANCED: Advanced audio analysis and voice detection
├── train_visual.py           # 🔥 ENHANCED: Training script with multimodal support
├── test_visual.py            # 🔥 ENHANCED: Comprehensive testing and analysis
├── requirements.txt          # All dependencies
├── setup_project.py          # Project initialization
├── download_pretrained.py    # Download pretrained models
├── dataset/
│   ├── real_videos/         # Legitimate video samples
│   ├── scam_videos/         # Scam video samples
│   └── processed_frames/    # Cached frame extractions
└── models/
    └── weights/             # Saved model weights
```

## 🛠️ Installation

1. **Clone and Setup Environment**
```bash
cd Multimodals
pip install -r requirements.txt
```

2. **Setup Project Structure**
```bash
python setup_project.py
```

3. **Download Pretrained Models** (Optional)
```bash
python download_pretrained.py
```

4. **Add Training Data**
   - Place legitimate videos in `dataset/real_videos/`
   - Place scam videos in `dataset/scam_videos/`

## 🎯 Usage

### Quick Analysis
```bash
# Simple analysis
python test_visual.py path/to/video.mp4

# Enhanced feature testing  
python test_visual.py path/to/video.mp4 --enhanced
```

### Programmatic Usage
```python
from visual_engine import analyze_video_quick, VisualQualityHead

# Quick analysis - INTEGRATED ENHANCED FUNCTION
result = analyze_video_quick('video.mp4')
print(f"Credibility Score: {result['overall_score']:.4f}")
print(f"Verdict: {result['verdict']}")

# Detailed analysis with enhanced model
model = VisualQualityHead()
model.load_head_weights('models/weights/enhanced_visual_model.pth')
detailed_result = model.analyze_video_complete('video.mp4')
```

### Training the System
```bash
# Multimodal training with enhanced features
python train_visual.py
```

## 📊 Analysis Output

The enhanced system provides comprehensive analysis:

### Overall Assessment
- **Credibility Score**: 0.0 (Scam) to 1.0 (Legitimate)
- **Confidence Level**: System confidence in the prediction
- **Risk Level**: HIGH/MEDIUM/LOW risk categorization
- **Final Verdict**: Human-readable assessment

### Visual Analysis
- **Production Quality**: Professional vs amateur assessment
- **Temporal Consistency**: Frame-to-frame coherence
- **Lighting Analysis**: Professional lighting indicators
- **Background Integrity**: Compositing artifact detection

### Audio Analysis
- **Voice Authenticity**: Natural vs synthetic voice detection
- **Emotional Patterns**: Scripted vs natural emotional expression
- **Speech Characteristics**: Rate, pauses, and naturalness
- **Technical Metrics**: Jitter, shimmer, pitch stability

### Example Output
```
ANALYSIS RESULTS
================
Overall Score: 0.234
Confidence: 0.87
Verdict: SCAM / LOW EFFORT
Risk Level: HIGH

MODALITY BREAKDOWN
==================
Visual Quality: 0.156
Audio Authenticity: 0.312

VOICE ANALYSIS
==============
Authenticity Score: 0.312
Voice Consistency: 0.445
Pitch Stability: 0.678
Likely Synthetic: False

EMOTIONAL ANALYSIS
==================
Emotional Authenticity: 0.234
Energy Variation: 0.567
Pitch Variation: 0.123
Speech Naturalness: 0.445
Likely Scripted: True
```

## 🧠 Technical Architecture

### Visual Pipeline
1. **Frame Extraction**: Smart sampling across video timeline
2. **MobileNetV2 Backbone**: Frozen pretrained feature extraction
3. **Temporal Analyzer**: LSTM + Attention for sequence analysis
4. **Quality Assessment**: Multi-factor production quality scoring

### Audio Pipeline
1. **VGGish Embeddings**: Pretrained audio feature extraction
2. **Voice Analysis**: Pitch tracking, jitter/shimmer calculation
3. **Emotional Analysis**: Energy dynamics and speech patterns
4. **Temporal Features**: Consistency and authenticity metrics

### Fusion Strategy
1. **Feature Preprocessing**: Modality-specific normalization
2. **Cross-Modal Attention**: Learn inter-modal relationships
3. **Adaptive Weighting**: Dynamic importance balancing
4. **Final Classification**: Multi-layer decision network

## 🔧 Advanced Configuration

### Custom Model Training
```python
# ENHANCED MULTIMODAL SYSTEM - Complete integrated training
from visual_engine import VisualQualityHead

model = VisualQualityHead()

# Enhanced training with temporal & fusion components
# Use: python train_visual.py

# Programmatic training setup
import torch
import torch.optim as optim

# Setup optimizers for different components
basic_optimizer = optim.Adam(model.head.parameters(), lr=0.001)
temporal_optimizer = optim.Adam(
    list(model.temporal_lstm.parameters()) + 
    list(model.temporal_attention.parameters()), 
    lr=0.0005
)
fusion_optimizer = optim.Adam(model.fusion_network.parameters(), lr=0.0005)

# Save trained models
model.save_head_weights('models/weights/enhanced_visual_model.pth')
```

### Individual Component Analysis
```python
# INTEGRATED SYSTEM - All in visual_engine.py
from visual_engine import VisualQualityHead
from audio_engine import analyze_audio_complete

model = VisualQualityHead()

# Analyze just visual components
visual_features = model.extract_comprehensive_visual_features('video.mp4')

# Analyze just audio components  
audio_results = analyze_audio_complete('video.mp4')

# Combined multimodal analysis
complete_results = model.analyze_video_complete('video.mp4')
```

## 📈 Performance Characteristics

### Strengths
- **Multi-modal robustness**: Harder to fool multiple detection systems
- **Temporal awareness**: Detects editing artifacts and inconsistencies
- **Voice authenticity**: Advanced vocal pattern analysis
- **Production quality**: Distinguishes professional from amateur content
- **Confidence scoring**: Reliable uncertainty quantification

### Limitations
- **Computational cost**: Requires significant processing power
- **Training data**: Performance depends on quality/quantity of training samples
- **Language dependency**: Voice analysis optimized for English
- **Video quality**: Very low quality videos may reduce accuracy

### Recommended Use Cases
- **Social media verification**: Influencer authenticity assessment
- **Financial content**: Investment scam detection
- **News verification**: Deepfake and manipulated content detection
- **E-commerce**: Product review authenticity
- **Educational content**: Academic integrity verification

## 🛡️ Security Considerations

- The system is designed for detection, not absolute proof
- Always combine with human expert review for critical decisions
- Regular retraining recommended as scam techniques evolve
- Consider privacy implications when analyzing personal content

## 🔮 Future Enhancements

- **Real-time analysis**: Live stream processing capabilities
- **Multi-language support**: Extended voice analysis for other languages
- **Blockchain verification**: Immutable authenticity certificates
- **Advanced deepfake detection**: State-of-the-art synthetic media detection
- **Behavioral analysis**: User interaction pattern assessment

## 📞 Support

For technical questions or issues:
1. Check the test output for diagnostic information
2. Verify all dependencies are installed correctly
3. Ensure video files are in supported formats (MP4, AVI, MOV)
4. Monitor system resources during analysis

The enhanced system represents a significant advancement in multimodal authenticity detection, combining cutting-edge deep learning with practical deployment considerations.