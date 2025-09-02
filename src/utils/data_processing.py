"""
Data processing utilities for Project Springfield
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path

class DataProcessor:
    """Handles data processing and analysis tasks"""
    
    def __init__(self):
        self.court_length = 94  # feet
        self.court_width = 50   # feet
        
    def normalize_coordinates(self, 
                            coordinates: List[Tuple[float, float]],
                            frame_dimensions: Tuple[int, int],
                            court_dimensions: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
        """
        Normalize pixel coordinates to court coordinates
        
        Args:
            coordinates: List of (x, y) pixel coordinates
            frame_dimensions: (width, height) of the frame
            court_dimensions: (length, width) of the court in feet
        """
        if court_dimensions is None:
            court_dimensions = (self.court_length, self.court_width)
        
        frame_width, frame_height = frame_dimensions
        court_length, court_width = court_dimensions
        
        normalized = []
        for x, y in coordinates:
            norm_x = (x / frame_width) * court_length
            norm_y = (y / frame_height) * court_width
            normalized.append((norm_x, norm_y))
        
        return normalized
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_speed(self, 
                       trajectory: List[Tuple[float, float, float]],  # (x, y, timestamp)
                       smoothing_window: int = 5) -> List[float]:
        """Calculate speed from trajectory data"""
        if len(trajectory) < 2:
            return []
        
        speeds = []
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            distance = self.calculate_distance(
                (prev_point[0], prev_point[1]), 
                (curr_point[0], curr_point[1])
            )
            time_diff = curr_point[2] - prev_point[2]
            
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
            else:
                speeds.append(0.0)
        
        # Apply smoothing
        if smoothing_window > 1:
            speeds = self.smooth_data(speeds, smoothing_window)
        
        return speeds
    
    def smooth_data(self, data: List[float], window_size: int) -> List[float]:
        """Apply moving average smoothing to data"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def detect_possession_changes(self, 
                                 player_positions: Dict[int, List[Tuple[float, float]]],
                                 ball_positions: List[Tuple[float, float]],
                                 threshold_distance: float = 5.0) -> List[Dict]:
        """Detect possession changes based on proximity"""
        possessions = []
        current_possessor = None
        
        for frame_idx, ball_pos in enumerate(ball_positions):
            closest_player = None
            closest_distance = float('inf')
            
            # Find closest player to ball
            for player_id, positions in player_positions.items():
                if frame_idx < len(positions):
                    player_pos = positions[frame_idx]
                    distance = self.calculate_distance(ball_pos, player_pos)
                    
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_player = player_id
            
            # Check for possession change
            if (closest_player is not None and 
                closest_distance < threshold_distance and 
                closest_player != current_possessor):
                
                possessions.append({
                    'frame': frame_idx,
                    'player_id': closest_player,
                    'previous_possessor': current_possessor,
                    'distance': closest_distance
                })
                current_possessor = closest_player
        
        return possessions
    
    def export_to_json(self, data: Dict[str, Any], output_path: str):
        """Export data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def export_to_csv(self, data: Dict[str, Any], output_path: str):
        """Export data to CSV file"""
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def load_annotations(self, annotation_path: str) -> Dict[str, Any]:
        """Load ground truth annotations"""
        with open(annotation_path, 'r') as f:
            return json.load(f)
    
    def calculate_basic_stats(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic basketball statistics"""
        stats = {
            'total_frames': len(game_data.get('frames', [])),
            'players_detected': len(set(game_data.get('player_ids', []))),
            'possession_changes': len(game_data.get('possessions', [])),
            'shots_detected': len(game_data.get('shots', [])),
        }
        
        # Calculate per-player stats
        player_stats = {}
        for player_id in set(game_data.get('player_ids', [])):
            player_shots = [s for s in game_data.get('shots', []) 
                          if s.get('player_id') == player_id]
            
            made_shots = [s for s in player_shots if s.get('made', False)]
            
            player_stats[player_id] = {
                'shots_attempted': len(player_shots),
                'shots_made': len(made_shots),
                'fg_percentage': len(made_shots) / len(player_shots) if player_shots else 0,
            }
        
        stats['player_stats'] = player_stats
        return stats