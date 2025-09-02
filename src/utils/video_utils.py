# video generator 09/25
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Union
from tqdm import tqdm

class VideoProcessor:
    # handles processing 09/25
    
    def __init__(self, config_manager=None):
        self.config = config_manager.get_pipeline_config() if config_manager else {}
        self.video_config = self.config.get("video_processing", {})
    
    def read_video_frames(self, 
                         video_path: Union[str, Path], 
                         start_frame: int = 0,
                         end_frame: Optional[int] = None,
                         skip_frames: int = 0) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Generator to read video frames
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)
            skip_frames: Number of frames to skip between reads
            
        Yields:
            Tuple of (frame, frame_number)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video Info: {total_frames} frames, {fps:.2f} FPS")
        
        # Set starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_number = start_frame
        end_frame = end_frame or total_frames
        
        with tqdm(total=min(end_frame - start_frame, total_frames - start_frame), 
                  desc="Processing frames") as pbar:
            
            while frame_number < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize if specified
                if self.video_config.get("resize_height") and self.video_config.get("resize_width"):
                    frame = cv2.resize(frame, 
                                     (self.video_config["resize_width"], 
                                      self.video_config["resize_height"]))
                
                yield frame, frame_number
                
                # Skip frames if specified
                if skip_frames > 0:
                    for _ in range(skip_frames):
                        ret, _ = cap.read()
                        if not ret:
                            break
                        frame_number += 1
                
                frame_number += 1
                pbar.update(1)
        
        cap.release()
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def create_video_writer(self, 
                           output_path: Union[str, Path],
                           fps: float,
                           frame_size: Tuple[int, int],
                           codec: str = "mp4v") -> cv2.VideoWriter:
        """Create a video writer object"""
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        
        if not writer.isOpened():
            raise ValueError(f"Could not create video writer for: {output_path}")
        
        return writer
    
    @staticmethod
    def extract_frames(video_path: Union[str, Path], 
                      output_dir: Union[str, Path],
                      max_frames: Optional[int] = None,
                      frame_interval: int = 1) -> int:
        """Extract frames from video to images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = output_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
                
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return saved_count