"""
Ball Detection Module for Project Springfield
Basketball detection using YOLOv8
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

class BallDetector:
    """
    Basketball detector using YOLOv8
    Detects basketballs on court and returns bounding boxes with confidence scores
    """
    
    def __init__(self, config_manager=None, model_path: Optional[str] = None):
        """
        Initialize the BallDetector
        
        Args:
            config_manager: ConfigManager instance with model configurations
            model_path: Optional path to custom trained model
        """
        self.config_manager = config_manager
        self.model = None
        self.device = "cpu"  # Default fallback
        self.confidence_threshold = 0.3  # Lower for ball detection (harder to detect)
        self.iou_threshold = 0.5
        self.input_size = (640, 640)
        
        # COCO class ID for sports ball (basketball)
        self.sports_ball_class_id = 32  # Sports ball in COCO dataset
        
        # Ball tracking for trajectory smoothing
        self.previous_detections = []
        self.max_history = 10
        
        # Load configuration
        self._load_config()
        
        # Initialize model
        self._initialize_model(model_path)
        
    def _load_config(self):
        """Load configuration from config manager"""
        if self.config_manager:
            try:
                ball_config = self.config_manager.get_model_config("ball_detection")
                self.confidence_threshold = ball_config.get("confidence_threshold", 0.3)
                self.iou_threshold = ball_config.get("iou_threshold", 0.5)
                self.input_size = tuple(ball_config.get("input_size", [640, 640]))
                
                # Set device with fallback
                device_config = ball_config.get("device", "cpu")
                if device_config == "cuda" and torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info(f"Ball detector using GPU: {torch.cuda.get_device_name()}")
                else:
                    self.device = "cpu"
                    logger.info("Ball detector using CPU for inference")
                    
            except Exception as e:
                logger.warning(f"Could not load ball detection config: {e}. Using defaults.")
    
    def _initialize_model(self, model_path: Optional[str] = None):
        """Initialize YOLOv8 model for ball detection"""
        try:
            if model_path and Path(model_path).exists():
                # Load custom trained model
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom ball detection model: {model_path}")
            else:
                # Load pretrained YOLOv8 model (usually better for ball detection)
                model_name = "yolov8s.pt"  # Default to small for better accuracy
                if self.config_manager:
                    try:
                        ball_config = self.config_manager.get_model_config("ball_detection")
                        model_name = ball_config.get("name", "yolov8s") + ".pt"
                    except:
                        pass
                
                self.model = YOLO(model_name)
                logger.info(f"Loaded pretrained model for ball detection: {model_name}")
            
            # Move model to appropriate device
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to initialize ball detection model: {e}")
            raise
    
    def detect(self, frame: np.ndarray, use_tracking: bool = True) -> List[Dict]:
        """
        Detect basketballs in a single frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            use_tracking: Whether to use trajectory smoothing
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
            - class_id: Class ID (32 for sports ball)
            - class_name: "basketball"
        """
        if self.model is None:
            raise RuntimeError("Ball detection model not initialized")
        
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
                    
                    # Filter for sports ball class only
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        if cls_id == self.sports_ball_class_id:  # Sports ball class
                            detection = {
                                'bbox': box.tolist(),  # [x1, y1, x2, y2]
                                'confidence': float(conf),
                                'class_id': cls_id,
                                'class_name': 'basketball'
                            }
                            detections.append(detection)
            
            # Apply trajectory smoothing if enabled
            if use_tracking and detections:
                detections = self._smooth_detections(detections)
            
            # Store for next frame
            self.previous_detections.append(detections)
            if len(self.previous_detections) > self.max_history:
                self.previous_detections.pop(0)
            
            logger.debug(f"Detected {len(detections)} basketballs")
            return detections
            
        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            return []
    
    def _smooth_detections(self, current_detections: List[Dict]) -> List[Dict]:
        """Apply trajectory smoothing to reduce noise in ball detection"""
        if not self.previous_detections or not current_detections:
            return current_detections
        
        smoothed_detections = []
        
        for detection in current_detections:
            bbox = detection['bbox']
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            
            # Find closest detection from previous frame
            if self.previous_detections:
                prev_detections = self.previous_detections[-1]
                if prev_detections:
                    closest_prev = self._find_closest_detection(center, prev_detections)
                    
                    if closest_prev:
                        # Apply smoothing to reduce jitter
                        alpha = 0.7  # Smoothing factor
                        prev_bbox = closest_prev['bbox']
                        prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, 
                                     (prev_bbox[1] + prev_bbox[3]) / 2]
                        
                        # Smooth the center coordinates
                        smooth_center = [
                            alpha * center[0] + (1 - alpha) * prev_center[0],
                            alpha * center[1] + (1 - alpha) * prev_center[1]
                        ]
                        
                        # Update bbox with smoothed center
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        
                        detection['bbox'] = [
                            smooth_center[0] - width/2,
                            smooth_center[1] - height/2,
                            smooth_center[0] + width/2,
                            smooth_center[1] + height/2
                        ]
            
            smoothed_detections.append(detection)
        
        return smoothed_detections
    
    def _find_closest_detection(self, center: List[float], prev_detections: List[Dict]) -> Optional[Dict]:
        """Find the closest detection from previous frame"""
        min_distance = float('inf')
        closest_detection = None
        
        for prev_detection in prev_detections:
            prev_bbox = prev_detection['bbox']
            prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, 
                          (prev_bbox[1] + prev_bbox[3]) / 2]
            
            distance = np.sqrt((center[0] - prev_center[0])**2 + 
                             (center[1] - prev_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_detection = prev_detection
        
        # Only return if distance is reasonable (ball doesn't teleport)
        if min_distance < 100:  # pixels
            return closest_detection
        return None
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect basketballs in multiple frames
        
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
    
    def get_ball_trajectory(self, n_frames: int = 10) -> List[Tuple[float, float]]:
        """
        Get recent ball trajectory as list of center points
        
        Args:
            n_frames: Number of recent frames to include
            
        Returns:
            List of (x, y) center coordinates
        """
        trajectory = []
        
        # Get recent frames (up to n_frames)
        recent_frames = self.previous_detections[-n_frames:] if self.previous_detections else []
        
        for frame_detections in recent_frames:
            if frame_detections:
                # Take the first (most confident) detection
                detection = frame_detections[0]
                bbox = detection['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                trajectory.append(center)
        
        return trajectory
    
    def reset_tracking(self):
        """Reset tracking history (useful when switching videos)"""
        self.previous_detections = []
        logger.info("Ball tracking history reset")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'input_size': self.input_size,
            'model_loaded': self.model is not None,
            'tracking_enabled': True,
            'max_history': self.max_history
        }
        
        if self.model:
            info['model_type'] = str(type(self.model).__name__)
        
        return info
