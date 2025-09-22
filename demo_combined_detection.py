#!/usr/bin/env python3
"""
Project Springfield - Combined Basketball Detection Demo
Tests both player and ball detection working together
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config_manager import ConfigManager
from src.detection.player_detector import PlayerDetector
from src.detection.ball_detector import BallDetector
from src.utils.visualization import Visualizer

def create_basketball_test_scene():
    """Create a test scene with players and ball for detection testing"""
    # Create basketball court background
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 30  # Dark court
    
    # Draw court markings
    # Court boundary
    cv2.rectangle(img, (100, 100), (1180, 620), (255, 255, 255), 3)
    # Center line
    cv2.line(img, (640, 100), (640, 620), (255, 255, 255), 2)
    # Center circle
    cv2.circle(img, (640, 360), 80, (255, 255, 255), 2)
    # Free throw circles
    cv2.circle(img, (240, 360), 60, (255, 255, 255), 2)
    cv2.circle(img, (1040, 360), 60, (255, 255, 255), 2)
    
    # Add players (larger, more realistic)
    # Player 1 - Red team
    cv2.rectangle(img, (200, 280), (260, 420), (0, 0, 200), -1)  # Body
    cv2.circle(img, (230, 260), 25, (255, 220, 180), -1)  # Head
    cv2.putText(img, "23", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Player 2 - Blue team  
    cv2.rectangle(img, (400, 300), (460, 440), (200, 100, 0), -1)  # Body
    cv2.circle(img, (430, 280), 25, (255, 220, 180), -1)  # Head
    cv2.putText(img, "7", (425, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Player 3 - Red team
    cv2.rectangle(img, (800, 250), (860, 390), (0, 0, 200), -1)  # Body
    cv2.circle(img, (830, 230), 25, (255, 220, 180), -1)  # Head
    cv2.putText(img, "12", (820, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Player 4 - Blue team
    cv2.rectangle(img, (950, 320), (1010, 460), (200, 100, 0), -1)  # Body
    cv2.circle(img, (980, 300), 25, (255, 220, 180), -1)  # Head
    cv2.putText(img, "33", (970, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add basketball (orange circle with black lines)
    ball_center = (580, 200)
    cv2.circle(img, ball_center, 15, (0, 165, 255), -1)  # Orange ball
    # Basketball lines
    cv2.line(img, (ball_center[0]-12, ball_center[1]), (ball_center[0]+12, ball_center[1]), (0, 0, 0), 2)
    cv2.line(img, (ball_center[0], ball_center[1]-12), (ball_center[0], ball_center[1]+12), (0, 0, 0), 2)
    # Curved lines
    cv2.ellipse(img, ball_center, (12, 8), 45, 0, 180, (0, 0, 0), 1)
    cv2.ellipse(img, ball_center, (12, 8), -45, 0, 180, (0, 0, 0), 1)
    
    return img

def demo_combined_detection(player_detector: PlayerDetector, 
                           ball_detector: BallDetector, 
                           visualizer: Visualizer):
    """Demo showing both player and ball detection"""
    print("🏀 Combined Basketball Detection Demo")
    print("=" * 50)
    
    # Create test scene
    test_scene = create_basketball_test_scene()
    
    print("🔍 Running detections...")
    
    # Detect players
    player_detections = player_detector.detect(test_scene)
    print(f"👥 Found {len(player_detections)} players")
    
    # Detect ball
    ball_detections = ball_detector.detect(test_scene)
    print(f"🏀 Found {len(ball_detections)} basketballs")
    
    # Combine detections
    all_detections = player_detections + ball_detections
    
    # Create custom class names for visualization
    class_names = {
        0: "Player",      # Person class
        32: "Basketball"  # Sports ball class
    }
    
    # Annotate image
    annotated_scene = visualizer.draw_detections(
        test_scene,
        all_detections,
        class_names=class_names,
        show_confidence=True
    )
    
    # Display results
    cv2.namedWindow('Project Springfield - Combined Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Project Springfield - Combined Detection', 1280, 720)
    cv2.imshow('Project Springfield - Combined Detection', annotated_scene)
    
    print("📊 Detection Results:")
    print(f"   Players detected: {len(player_detections)}")
    print(f"   Basketballs detected: {len(ball_detections)}")
    print(f"   Total detections: {len(all_detections)}")
    
    # Print detailed results
    for i, detection in enumerate(player_detections):
        bbox = detection['bbox']
        conf = detection['confidence']
        print(f"   Player {i+1}: confidence={conf:.3f}, bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    for i, detection in enumerate(ball_detections):
        bbox = detection['bbox']
        conf = detection['confidence']
        print(f"   Ball {i+1}: confidence={conf:.3f}, bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    print("\n✨ Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return annotated_scene, all_detections

def demo_webcam_combined(player_detector: PlayerDetector, 
                        ball_detector: BallDetector, 
                        visualizer: Visualizer):
    """Combined detection demo using webcam"""
    print("🎥 Combined Webcam Detection Demo")
    print("Press 'q' to quit, 's' to save frame, 'r' to reset ball tracking")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players and ball
            player_detections = player_detector.detect(frame)
            ball_detections = ball_detector.detect(frame)
            
            # Combine detections
            all_detections = player_detections + ball_detections
            
            # Class names for visualization
            class_names = {0: "Player", 32: "Basketball"}
            
            # Annotate frame
            annotated_frame = visualizer.draw_detections(
                frame,
                all_detections,
                class_names=class_names,
                show_confidence=True
            )
            
            # Add ball trajectory if available
            trajectory = ball_detector.get_ball_trajectory(10)
            if len(trajectory) > 1:
                # Draw trajectory
                for i in range(1, len(trajectory)):
                    pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                    pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 2)  # Yellow trail
            
            # Add info overlay
            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            info_text = f"FPS: {fps:.1f} | Players: {len(player_detections)} | Balls: {len(ball_detections)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_text2 = "Press 'q'=quit, 's'=save, 'r'=reset tracking"
            cv2.putText(annotated_frame, info_text2, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Project Springfield - Live Combined Detection', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"combined_detection_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"💾 Saved frame to {filename}")
            elif key == ord('r'):
                ball_detector.reset_tracking()
                print("🔄 Ball tracking reset")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Processed {frame_count} frames at {fps:.1f} FPS average")

def performance_test(player_detector: PlayerDetector, ball_detector: BallDetector):
    """Test performance of combined detection"""
    print("⚡ Combined Detection Performance Test")
    
    test_scene = create_basketball_test_scene()
    
    # Test player detection
    n_runs = 20
    player_times = []
    ball_times = []
    combined_times = []
    
    print(f"Running {n_runs} iterations...")
    
    for i in range(n_runs):
        # Test player detection alone
        start_time = time.time()
        player_detections = player_detector.detect(test_scene)
        player_time = time.time() - start_time
        player_times.append(player_time)
        
        # Test ball detection alone
        start_time = time.time()
        ball_detections = ball_detector.detect(test_scene)
        ball_time = time.time() - start_time
        ball_times.append(ball_time)
        
        # Test combined (sequential)
        start_time = time.time()
        player_detections = player_detector.detect(test_scene)
        ball_detections = ball_detector.detect(test_scene)
        combined_time = time.time() - start_time
        combined_times.append(combined_time)
        
        if i % 5 == 0:
            print(f"  Completed {i+1}/{n_runs} runs...")
    
    # Calculate averages
    avg_player_time = np.mean(player_times) * 1000  # ms
    avg_ball_time = np.mean(ball_times) * 1000  # ms
    avg_combined_time = np.mean(combined_times) * 1000  # ms
    
    combined_fps = 1.0 / np.mean(combined_times)
    
    print(f"\n📊 Performance Results:")
    print(f"   Player detection: {avg_player_time:.1f} ms")
    print(f"   Ball detection: {avg_ball_time:.1f} ms")  
    print(f"   Combined detection: {avg_combined_time:.1f} ms")
    print(f"   Combined FPS: {combined_fps:.1f}")
    
    if combined_fps < 15:
        print("⚠️ Performance below real-time threshold (15 FPS)")
        print("💡 Consider: GPU acceleration, smaller models, or parallel processing")
    else:
        print("✅ Performance suitable for real-time basketball analysis!")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Project Springfield Combined Detection Demo")
    parser.add_argument("--mode", choices=["test", "webcam", "performance"], default="test",
                       help="Demo mode")
    parser.add_argument("--config", type=str, default="configs", help="Config directory")
    
    args = parser.parse_args()
    
    print("🏀 Project Springfield - Combined Basketball Detection")
    print("=" * 60)
    
    try:
        # Initialize configuration
        print("⚙️ Loading configuration...")
        config_manager = ConfigManager(args.config)
        
        # Initialize detectors
        print("🤖 Initializing player detector...")
        player_detector = PlayerDetector(config_manager)
        
        print("🏀 Initializing ball detector...")
        ball_detector = BallDetector(config_manager)
        
        # Print model info
        player_info = player_detector.get_model_info()
        ball_info = ball_detector.get_model_info()
        
        print(f"\n📋 Player Detector Info:")
        for key, value in player_info.items():
            print(f"   {key}: {value}")
            
        print(f"\n🏀 Ball Detector Info:")
        for key, value in ball_info.items():
            print(f"   {key}: {value}")
        
        # Initialize visualizer
        visualizer = Visualizer()
        
        print("\n🚀 Starting combined detection demo...")
        print("-" * 60)
        
        # Run appropriate demo
        if args.mode == "test":
            demo_combined_detection(player_detector, ball_detector, visualizer)
        elif args.mode == "webcam":
            demo_webcam_combined(player_detector, ball_detector, visualizer)
        elif args.mode == "performance":
            performance_test(player_detector, ball_detector)
        
        print("🎉 Combined detection demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
