"""
Player Detection Module for Project Springfield
Basketball player detection using YOLOv8
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerDetector:
    """
    Basketball player detector using YOLOv8
    Detects players on basketball court and returns bounding boxes with confidence scores
    """
    
    def __init__(self, config_manager=None, model_path: Optional[str] = None):
        """
        Initialize the PlayerDetector
        
        Args:
            config_manager: ConfigManager instance with model configurations
            model_path: Optional path to custom trained model
        """
        self.config_manager = config_manager
        self.model = None
        self.device = "cpu"  # Default fallback
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        self.input_size = (640, 640)
        
        # COCO class ID for person (players)
        self.person_class_id = 0
        
        # Load configuration
        self._load_config()
        
        # Initialize model
        self._initialize_model(model_path)
        
    def _load_config(self):
        """Load configuration from config manager"""
        if self.config_manager:
            try:
                player_config = self.config_manager.get_model_config("player_detection")
                self.confidence_threshold = player_config.get("confidence_threshold", 0.5)
                self.iou_threshold = player_config.get("iou_threshold", 0.4)
                self.input_size = tuple(player_config.get("input_size", [640, 640]))
                
                # Set device with fallback
                device_config = player_config.get("device", "cpu")
                if device_config == "cuda" and torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                else:
                    self.device = "cpu"
                    logger.info("Using CPU for inference")
                    
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using defaults.")
    
    def _initialize_model(self, model_path: Optional[str] = None):
        """Initialize YOLOv8 model"""
        try:
            if model_path and Path(model_path).exists():
                # Load custom trained model
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom model: {model_path}")
            else:
                # Load pretrained YOLOv8 model
                model_name = "yolov8n.pt"  # Default to nano for speed
                if self.config_manager:
                    try:
                        player_config = self.config_manager.get_model_config("player_detection")
                        model_name = player_config.get("name", "yolov8n") + ".pt"
                    except:
                        pass
                
                self.model = YOLO(model_name)
                logger.info(f"Loaded pretrained model: {model_name}")
            
            # Move model to appropriate device
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
            - class_id: Class ID (0 for person/player)
            - class_name: "player"
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Filter for person class only (basketball players)
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        if cls_id == self.person_class_id:  # Person class
                            detection = {
                                'bbox': box.tolist(),  # [x1, y1, x2, y2]
                                'confidence': float(conf),
                                'class_id': cls_id,
                                'class_name': 'player'
                            }
                            detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} players")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect players in multiple frames
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            List of detection lists, one for each frame
        """
        results = []
        for frame in frames:
            detections = self.detect(frame)
            results.append(detections)
        return results
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> List[List[Dict]]:
        """
        Detect players in entire video
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            
        Returns:
            List of detection lists, one for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {total_frames} frames at {fps:.1f} FPS")
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect players
                detections = self.detect(frame)
                all_detections.append(detections)
                
                # Annotate and save frame if requested
                if writer is not None:
                    annotated_frame = self._annotate_frame(frame, detections)
                    writer.write(annotated_frame)
                
                frame_number += 1
                if frame_number % 30 == 0:  # Log progress every 30 frames
                    logger.info(f"Processed {frame_number}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        logger.info(f"Completed processing {len(all_detections)} frames")
        return all_detections
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Annotate frame with detection bounding boxes
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw labl
            label = f"Player {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'input_size': self.input_size,
            'model_loaded': self.model is not None
        }
        
        if self.model:
            info['model_type'] = str(type(self.model).__name__)
        
        return info
