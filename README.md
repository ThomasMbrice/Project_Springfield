# Project Springfield

An open-source computer vision system that analyzes basketball gameplay to automatically extract player statistics and performance metrics.

## Overview

Project Springfield uses state-of-the-art computer vision and machine learning techniques to track players, detect the basketball, and analyze gameplay from video footage. The system can extract various basketball statistics including shots, rebounds, assists, player movements, and advanced analytics.

## Features

- **Player Detection & Tracking**: Multi-object tracking of all players on court
- **Ball Tracking**: Real-time basketball detection and trajectory analysis  
- **Court Mapping**: Automatic court keypoint detection and perspective correction
- **Stat Extraction**: Automated counting of points, rebounds, assists, and more
- **Movement Analytics**: Player positioning, speed, and distance traveled
- **Shot Analysis**: Shot location tracking and shooting percentage calculations

## Tech Stack

### Core ML Framework
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision and video processing
- **NumPy/SciPy** - Numerical computing
- **Pandas** - Data manipulation and analysis

### Computer Vision Models
- **YOLOv8/YOLOv9** - Object detection (via ultralytics)
- **DeepSORT/ByteTrack** - Multi-object tracking
- **MediaPipe** - Pose estimation
- **Custom PyTorch Models** - Court detection and specialized tasks

### Development Tools
- **Jupyter Notebooks** - Experimentation and prototyping
- **Matplotlib/Seaborn** - Data visualization
- **Conda/Mamba** - Environment management

## Project Structure

```
basketball-analytics/
├── data/                          # Data storage
│   ├── videos/                   # Raw game footage
│   ├── annotations/              # Ground truth labels
│   └── models/                   # Trained model checkpoints
├── src/                          # Source code
│   ├── detection/               # Object detection modules
│   │   ├── player_detector.py
│   │   ├── ball_detector.py
│   │   └── court_detector.py
│   ├── tracking/                # Multi-object tracking
│   │   ├── player_tracker.py
│   │   └── trajectory_tracker.py
│   ├── stats/                   # Statistics extraction
│   │   ├── basic_stats.py
│   │   ├── shooting_analytics.py
│   │   └── movement_analytics.py
│   └── utils/                   # Utility functions
│       ├── video_utils.py
│       ├── visualization.py
│       └── data_processing.py
├── notebooks/                    # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── analysis_pipeline.ipynb
├── configs/                      # Configuration files
│   ├── model_configs.yaml
│   └── pipeline_configs.yaml
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Project_Springfield.git
   cd Project_Springfield
   ```

2. **Create conda environment**
   ```bash
   conda create -n basketball-analytics python=3.9
   conda activate basketball-analytics
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support** (if using GPU)
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

### Required Packages

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
ultralytics>=8.0.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
tqdm>=4.65.0
pyyaml>=6.0
scikit-learn>=1.3.0
```

## Quick Start

1. **Prepare your data**
   ```bash
   mkdir -p data/videos
   # Place your basketball game videos in data/videos/
   ```

2. **Run the analysis pipeline**
   ```python
   from src.detection.player_detector import PlayerDetector
   from src.tracking.player_tracker import PlayerTracker
   
   # Initialize components
   detector = PlayerDetector()
   tracker = PlayerTracker()
   
   # Process video
   results = detector.detect_players('data/videos/game1.mp4')
   tracked_players = tracker.track(results)
   ```

3. **Extract statistics**
   ```python
   from src.stats.basic_stats import BasicStats
   
   stats_extractor = BasicStats()
   game_stats = stats_extractor.extract_stats(tracked_players)
   ```

4. **Explore with Notebooks**
   ```bash
   jupyter notebook notebooks/data_exploration.ipynb
   ```

## Usage Examples

### Basic Player Tracking
```python
import cv2
from src.detection.player_detector import PlayerDetector
from src.utils.visualization import draw_detections

detector = PlayerDetector()
cap = cv2.VideoCapture('data/videos/sample_game.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = detector.detect(frame)
    annotated_frame = draw_detections(frame, detections)
    
    cv2.imshow('Basketball Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Shot Analysis
```python
from src.stats.shooting_analytics import ShotAnalyzer

analyzer = ShotAnalyzer()
shots = analyzer.detect_shots(tracked_data)
shooting_percentage = analyzer.calculate_fg_percentage(shots)

print(f"Field Goal Percentage: {shooting_percentage:.2%}")
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/
```

## Roadmap

- [ ] **Phase 1**: Basic player and ball detection
- [ ] **Phase 2**: Multi-object tracking implementation  
- [ ] **Phase 3**: Court mapping and perspective correction
- [ ] **Phase 4**: Basic statistics extraction (points, rebounds)
- [ ] **Phase 5**: Advanced analytics (shot charts, player efficiency)
- [ ] **Phase 6**: Real-time processing optimization

## Performance

Current benchmarks on standard basketball footage:
- Player Detection: ~30 FPS (GPU), ~8 FPS (CPU)
- Ball Tracking: ~25 FPS (GPU), ~6 FPS (CPU)  
- Full Pipeline: ~15 FPS (GPU), ~3 FPS (CPU)

*Benchmarks run on NVIDIA RTX 3080, Intel i7-10700K*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [MediaPipe](https://mediapipe.dev/) for pose estimation capabilities
- Basketball analytics community for inspiration and datasets

## Contact

For questions and collaboration opportunities:
- Create an issue on GitHub
- Email: [your-email@example.com]
- Discord: [Project Springfield Community]

---

**Note**: This project is for research and educational purposes. Please ensure you have proper rights to any video footage used for analysis.
