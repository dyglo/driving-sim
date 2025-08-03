"""
YOLOv8-based object detection for autonomous driving simulation.
Detects vehicles, pedestrians, and traffic signs in video frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import logging

from ..utils.config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    VEHICLE_CLASSES, PERSON_CLASS, TRAFFIC_SIGN_CLASSES
)

class ObjectDetector:
    """
    YOLOv8-based object detector for autonomous driving scenarios.
    """
    
    def __init__(self, model_path: str = YOLO_MODEL):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
        
        # COCO class names mapping
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench'
        }
        
    def _load_model(self):
        """Load the YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            logging.info(f"Successfully loaded YOLOv8 model: {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detected objects by category
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        detections = {
            'vehicles': [],
            'pedestrians': [],
            'traffic_signs': [],
            'other': []
        }
        
        # Process results
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # Extract box information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Create detection object
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'unknown'),
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                }
                
                # Categorize detection
                if class_id in VEHICLE_CLASSES:
                    detections['vehicles'].append(detection)
                elif class_id in PERSON_CLASS:
                    detections['pedestrians'].append(detection)
                elif class_id in TRAFFIC_SIGN_CLASSES:
                    detections['traffic_signs'].append(detection)
                else:
                    detections['other'].append(detection)
        
        # Debug logging
        total_detections = sum(len(detections[key]) for key in detections.keys())
        if total_detections > 0:
            logging.info(f"Detected {total_detections} objects: {len(detections['vehicles'])} vehicles, {len(detections['pedestrians'])} pedestrians, {len(detections['traffic_signs'])} signs")
        else:
            logging.debug("No objects detected in current frame")
        
        return detections
    
    def estimate_distance(self, bbox: List[int], known_width: float = 1.8) -> float:
        """
        Estimate distance to object based on bounding box width.
        Simple estimation assuming known object width.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            known_width: Known real-world width of object in meters
            
        Returns:
            Estimated distance in meters
        """
        # Simple distance estimation (this is a rough approximation)
        # In practice, you'd need camera calibration parameters
        bbox_width = bbox[2] - bbox[0]
        
        if bbox_width <= 0:
            return float('inf')
        
        # Focal length approximation (would need camera calibration)
        focal_length = 700  # pixels (typical for dashcam)
        
        # Distance = (known_width * focal_length) / bbox_width_pixels
        distance = (known_width * focal_length) / bbox_width
        return max(distance, 1.0)  # Minimum 1 meter
    
    def filter_detections_by_roi(self, detections: Dict[str, List], 
                                roi_polygon: np.ndarray) -> Dict[str, List]:
        """
        Filter detections to only include those within a region of interest.
        
        Args:
            detections: Dictionary of detections
            roi_polygon: ROI polygon points
            
        Returns:
            Filtered detections
        """
        filtered = {key: [] for key in detections.keys()}
        
        for category, detection_list in detections.items():
            for detection in detection_list:
                center_point = tuple(detection['center'])
                
                # Check if center point is inside ROI
                if cv2.pointPolygonTest(roi_polygon, center_point, False) >= 0:
                    filtered[category].append(detection)
        
        return filtered
    
    def get_detection_summary(self, detections: Dict[str, List]) -> str:
        """
        Get a summary string of current detections.
        
        Args:
            detections: Dictionary of detections
            
        Returns:
            Summary string
        """
        total_vehicles = len(detections['vehicles'])
        total_pedestrians = len(detections['pedestrians'])
        total_signs = len(detections['traffic_signs'])
        
        return f"Vehicles: {total_vehicles} | Pedestrians: {total_pedestrians} | Signs: {total_signs}"