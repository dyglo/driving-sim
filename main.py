#!/usr/bin/env python3
"""
Autonomous Driving Simulation
Restored callback-based pipeline using VideoProcessor.
"""

import logging
from pathlib import Path
import time
import cv2

from src.detection.object_detector import ObjectDetector
from src.detection.lane_detector import LaneDetector
from src.visualization.visualizer import Visualizer
from src.decision.brain import AutonomousDrivingBrain
from src.utils.video_processor import VideoProcessor
from src.utils.config import INPUT_VIDEO, OUTPUT_VIDEO, SHOW_LIVE_FEED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_driving_sim.log'),
        logging.StreamHandler()
    ]
)

def process_frame(frame, context):
    """
    Process a single frame: detection, lane, decision, visualization.
    Context is a dict holding persistent objects and stats.
    """
    object_detector = context['object_detector']
    lane_detector = context['lane_detector']
    visualizer = context['visualizer']
    decision_brain = context['decision_brain']
    frame_count = context['frame_count']
    start_time = context['start_time']

    # Object detection
    detections = object_detector.detect_objects(frame)
    # Lane detection
    lane_data = lane_detector.detect_lanes(frame)
    # Calculate FPS and speed
    elapsed_time = time.time() - start_time
    current_fps = (frame_count + 1) / elapsed_time if elapsed_time > 0 else 0
    speed = 30 + (current_fps / 10)
    # Decision making
    decision_data = decision_brain.analyze_and_decide(detections, lane_data, speed, 0)
    # Enhanced visualization
    frame = visualizer.create_enhanced_dashboard_overlay(
        frame, detections, lane_data, current_fps, speed, decision_data.get('steering_angle', 0)
    )
    return frame

def progress_callback(progress_info):
    logging.info(
        f"Progress: {progress_info['progress_percent']:.1f}% "
        f"({progress_info['current_frame']}/{progress_info['total_frames']}) "
        f"Current FPS: {progress_info['fps_current']:.1f} "
        f"ETA: {progress_info['estimated_time_remaining']:.1f}s"
    )

def main():
    logging.info("Starting Autonomous Driving Simulation (Restored Pipeline)")
    logging.info(f"Input: {INPUT_VIDEO}")
    logging.info(f"Output: {OUTPUT_VIDEO}")
    try:
        object_detector = ObjectDetector()
        lane_detector = LaneDetector()
        visualizer = Visualizer()
        decision_brain = AutonomousDrivingBrain()
        logging.info("All components initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        return
    # Prepare context for process_frame
    context = {
        'object_detector': object_detector,
        'lane_detector': lane_detector,
        'visualizer': visualizer,
        'decision_brain': decision_brain,
        'frame_count': 0,
        'start_time': time.time(),
    }
    # Initialize VideoProcessor
    video_processor = VideoProcessor(str(INPUT_VIDEO), str(OUTPUT_VIDEO))
    # Run the simulation
    try:
        def frame_callback(frame):
            context['frame_count'] += 1
            return process_frame(frame, context)
        success = video_processor.process_video(
            frame_processor=frame_callback,
            progress_callback=progress_callback
        )
        if success:
            logging.info("Simulation completed successfully!")
            logging.info(f"Output saved to: {OUTPUT_VIDEO}")
        else:
            logging.error("Simulation failed!")
    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user")
    except Exception as e:
        logging.error(f"Simulation error: {e}")

if __name__ == "__main__":
    main()