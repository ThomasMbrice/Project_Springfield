# vizzz 09/25
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import colorsys

class Visualizer:
    """Handles all visualization tasks"""
    
    def __init__(self):
        # Define color palette for different objects
        self.colors = self._generate_colors(20)
        self.court_color = (0, 128, 0)  # Green
        self.ball_color = (0, 165, 255)  # Orange
        
    def _generate_colors(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def draw_detections(self, 
                       frame: np.ndarray,
                       detections: List[Dict],
                       class_names: Dict[int, str] = None,
                       show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries with 'bbox', 'class_id', 'confidence'
            class_names: Mapping from class_id to class name
            show_confidence: Whether to show confidence scores
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            class_id = detection.get('class_id', 0)
            confidence = detection.get('confidence', 1.0)
            track_id = detection.get('track_id', None)
            
            # Get color for this class/track
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Prepare label
            label_parts = []
            if class_names and class_id in class_names:
                label_parts.append(class_names[class_id])
            if track_id is not None:
                label_parts.append(f"ID:{track_id}")
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            label = " ".join(label_parts) if label_parts else f"Det"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame,
                         (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                         (int(bbox[0]) + label_size[0], int(bbox[1])),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def draw_trajectories(self, 
                         frame: np.ndarray,
                         trajectories: Dict[int, List[Tuple[int, int]]],
                         max_trail_length: int = 30) -> np.ndarray:
        """Draw object trajectories"""
        annotated_frame = frame.copy()
        
        for track_id, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue
                
            color = self.colors[track_id % len(self.colors)]
            
            # Draw trajectory line
            recent_points = trajectory[-max_trail_length:]
            for i in range(1, len(recent_points)):
                # Fade the trail
                alpha = i / len(recent_points)
                pt1 = recent_points[i-1]
                pt2 = recent_points[i]
                
                cv2.line(annotated_frame, pt1, pt2, color, 2)
        
        return annotated_frame
    
    def draw_court_overlay(self, 
                          frame: np.ndarray,
                          court_keypoints: List[Tuple[int, int]],
                          show_lines: bool = True) -> np.ndarray:
        """Draw basketball court overlay"""
        annotated_frame = frame.copy()
        
        if show_lines and len(court_keypoints) >= 4:
            # Draw court boundaries (simplified)
            pts = np.array(court_keypoints, np.int32)
            cv2.polylines(annotated_frame, [pts], True, self.court_color, 2)
        
        # Draw keypoints
        for point in court_keypoints:
            cv2.circle(annotated_frame, point, 5, self.court_color, -1)
        
        return annotated_frame
    
    def create_shot_chart(self, 
                         shots: List[Dict],
                         court_dimensions: Tuple[int, int] = (94, 50)) -> plt.Figure:
        """Create a shot chart visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw court outline (simplified)
        court_length, court_width = court_dimensions
        ax.set_xlim(0, court_length)
        ax.set_ylim(0, court_width)
        
        # Plot shots
        made_shots = [s for s in shots if s.get('made', False)]
        missed_shots = [s for s in shots if not s.get('made', False)]
        
        if made_shots:
            made_x = [s['location'][0] for s in made_shots]
            made_y = [s['location'][1] for s in made_shots]
            ax.scatter(made_x, made_y, c='green', s=50, alpha=0.7, label='Made')
        
        if missed_shots:
            missed_x = [s['location'][0] for s in missed_shots]
            missed_y = [s['location'][1] for s in missed_shots]
            ax.scatter(missed_x, missed_y, c='red', s=50, alpha=0.7, label='Missed')
        
        ax.set_title('Shot Chart')
        ax.set_xlabel('Court Length (ft)')
        ax.set_ylabel('Court Width (ft)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_stats_dashboard(self, stats: Dict[str, Any]) -> plt.Figure:
        """Create a statistics dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Basketball Analytics Dashboard', fontsize=16)
        
        # Plot 1: Shooting percentage by player
        if 'player_stats' in stats:
            players = list(stats['player_stats'].keys())
            fg_pcts = [stats['player_stats'][p].get('fg_percentage', 0) for p in players]
            
            axes[0, 0].bar(players, fg_pcts)
            axes[0, 0].set_title('Field Goal Percentage')
            axes[0, 0].set_ylabel('FG%')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Points distribution
        if 'scoring' in stats:
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            points = [stats['scoring'].get(q, 0) for q in quarters]
            
            axes[0, 1].plot(quarters, points, marker='o')
            axes[0, 1].set_title('Points by Quarter')
            axes[0, 1].set_ylabel('Points')
        
        # Plot 3: Movement heatmap (placeholder)
        axes[1, 0].imshow(np.random.random((10, 10)), cmap='hot')
        axes[1, 0].set_title('Player Movement Heatmap')
        
        # Plot 4: Shot distance distribution
        if 'shots' in stats:
            distances = [s.get('distance', 0) for s in stats['shots']]
            axes[1, 1].hist(distances, bins=20, alpha=0.7)
            axes[1, 1].set_title('Shot Distance Distribution')
            axes[1, 1].set_xlabel('Distance (ft)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig