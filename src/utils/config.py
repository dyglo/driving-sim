"""
Configuration settings for the autonomous driving simulation.
Enhanced with professional features and LinkedIn-ready settings.
"""

from pathlib import Path

# Directory structure
BASE_DIR = Path(__file__).parent.parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Input/Output files
INPUT_VIDEO = VIDEOS_DIR / "urbanroad.mp4"
OUTPUT_VIDEO = OUTPUT_DIR / "processed_output.mp4"

# Display settings
SHOW_LIVE_FEED = True  # Set to False for faster processing without display

# YOLOv8 Configuration
YOLO_MODEL = "yolov8n.pt"  # nano model for speed, use yolov8s.pt or yolov8m.pt for better accuracy
CONFIDENCE_THRESHOLD = 0.3  # Lowered from 0.5 to improve detection sensitivity
IOU_THRESHOLD = 0.45

# Classes of interest for autonomous driving (COCO dataset class IDs)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
PERSON_CLASS = [0]              # person/pedestrian
TRAFFIC_SIGN_CLASSES = [9, 10, 11, 12, 13]  # traffic light, fire hydrant, stop sign, parking meter, bench

# Lane Detection Configuration
LANE_DETECTION = {
    'canny_low': 50,
    'canny_high': 150,
    'hough_rho': 1,
    'hough_theta': 1.5707963267948966,  # π/2
    'hough_threshold': 50,
    'hough_min_line_length': 100,
    'hough_max_line_gap': 50
}

# Color scheme for professional visualization
COLORS = {
    'primary': (18, 28, 48),      # Dark blue
    'secondary': (60, 80, 120),    # Medium blue
    'accent': (0, 255, 255),       # Cyan
    'success': (0, 255, 0),        # Green
    'warning': (0, 255, 255),      # Yellow
    'danger': (0, 0, 255),         # Red
    'text': (255, 255, 255),       # White
    'text_secondary': (200, 200, 200),  # Light grey
}

# Video processing configuration
VIDEO_CONFIG = {
    'save_output': True,
    'show_progress': True,
    'max_fps': 30,  # Limit FPS for consistent processing
    'fps': 30,
    'codec': 'mp4v',
    'display_width': 1280,
    'display_height': 720,
    'show_realtime': True,
}

# Enhanced visualization settings
VISUALIZATION_CONFIG = {
    'enable_particle_effects': True,
    'enable_glassmorphism': True,
    'enable_safety_features': True,
    'enable_performance_metrics': True,
    'company_branding': "Tafar M",
}

# Performance settings
PERFORMANCE_CONFIG = {
    'target_fps': 30,
    'max_processing_time': 0.033,  # 30 FPS = 33ms per frame
    'enable_optimization': True,
    'skip_frames': 1,  # Process every nth frame (1 = process all)
    'resize_factor': 1.0,  # Resize input for processing speed
    'max_detections': 100,  # Maximum detections per frame
}