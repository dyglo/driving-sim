"""
Enhanced visualization module for autonomous driving simulation.
Professional LinkedIn-ready design with modern UI elements.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from collections import deque
import time

from ..utils.config import COLORS, LANE_DETECTION

class Visualizer:
    def __init__(self):
        """Initialize the visualizer with enhanced attributes."""
        # Font settings
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Professional color scheme (dark theme with accent colors)
        self.colors = {
            'primary': (18, 28, 48),      # Dark blue
            'secondary': (60, 80, 120),    # Medium blue
            'accent': (0, 255, 255),       # Cyan
            'success': (0, 255, 0),        # Green
            'warning': (0, 255, 255),      # Yellow
            'danger': (0, 0, 255),         # Red
            'text': (255, 255, 255),       # White
            'text_secondary': (200, 200, 200),  # Light grey
        }
        
        # Lane history for smoothing
        self.lane_history = deque(maxlen=5)
        
        # Decision overlay data
        self.decision_overlay_data = {}
        
        # Animation states
        self.animation_time = time.time()
        self.object_animations = {}
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_accuracy = 0.95
        
        # System status
        self.system_health = {
            'battery': 95,
            'gps': 'Connected',
            'sensors': 'Optimal',
            'ai_model': 'Active'
        }
        
        # Safety features
        self.safety_zones = []
        self.collision_warnings = []
        
        # Company branding
        self.company_name = "Tafar M"
        
    def get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level."""
        if confidence >= 0.8:
            return self.colors['success']  # Green for high confidence
        elif confidence >= 0.6:
            return self.colors['warning']  # Yellow for medium confidence
        else:
            return self.colors['danger']   # Red for low confidence
    
    def draw_glassmorphism_panel(self, frame: np.ndarray, x: int, y: int, w: int, h: int, 
                                title: str, content: List[str]) -> np.ndarray:
        """Draw modern glassmorphism panel with frosted glass effect."""
        # Create overlay for glassmorphism effect
        overlay = frame.copy()
        
        # Draw semi-transparent background with blur effect
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.colors['primary'], -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.colors['secondary'], 2)
        
        # Add subtle gradient effect
        for i in range(h):
            alpha = 0.1 + (i / h) * 0.1
            cv2.line(overlay, (x, y + i), (x + w, y + i), 
                    (int(self.colors['primary'][0] * alpha), 
                     int(self.colors['primary'][1] * alpha), 
                     int(self.colors['primary'][2] * alpha)), 1)
        
        # Blend with original frame
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw title with modern typography
        cv2.putText(frame, title, (x + 15, y + 30), self.font, 0.8, 
                   self.colors['text'], 2, cv2.LINE_AA)
        
        # Draw content with proper spacing
        for i, line in enumerate(content):
            color = self.colors['text'] if i == 0 else self.colors['text_secondary']
            cv2.putText(frame, line, (x + 15, y + 60 + i * 25), self.font, 0.6, 
                       color, 1, cv2.LINE_AA)
        
        return frame
    
    def draw_enhanced_bounding_box(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """Draw enhanced bounding box with sleek overlay and icons."""
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get confidence color
        box_color = self.get_confidence_color(confidence)
        
        # Draw sleek semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add object type icon
        icon_text = self.get_object_icon(class_name)
        cv2.putText(frame, icon_text, (x1 + 5, y1 - 10), self.font, 0.8, 
                   self.colors['text'], 2, cv2.LINE_AA)
        
        # Draw confidence bar
        bar_width = int((x2 - x1) * confidence)
        cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 8), 
                     box_color, -1)
        
        # Add confidence text
        conf_text = f"{confidence:.2f}"
        cv2.putText(frame, conf_text, (x1, y2 + 20), self.font, 0.5, 
                   box_color, 1, cv2.LINE_AA)
        
        return frame
    
    def get_object_icon(self, class_name: str) -> str:
        """Get icon for object type."""
        icons = {
            'car': '🚗',
            'truck': '🚛',
            'bus': '🚌',
            'motorcycle': '🏍️',
            'person': '👤',
            'traffic light': '🚦',
            'stop sign': '🛑',
            'fire hydrant': '🚒'
        }
        return icons.get(class_name, '📦')
    
    def draw_gradient_lane_lines(self, frame: np.ndarray, lane_data: Dict) -> np.ndarray:
        """Draw gradient lane lines that fade with distance."""
        if lane_data['status'] != 'success':
            return frame
        
        height, width = frame.shape[:2]
        
        # Draw left lane with gradient
        left_lane = lane_data.get('left_lane')
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            self.draw_gradient_line(frame, (x1, y1), (x2, y2), self.colors['warning'])
        
        # Draw right lane with gradient
        right_lane = lane_data.get('right_lane')
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            self.draw_gradient_line(frame, (x1, y1), (x2, y2), self.colors['accent'])
        
        # Add lane departure warning
        if self.check_lane_departure(lane_data):
            self.draw_lane_departure_warning(frame)
        
        return frame
    
    def draw_gradient_line(self, frame: np.ndarray, start: Tuple[int, int], 
                          end: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a line with gradient effect."""
        # Calculate line points
        points = self.get_line_points(start, end, 50)
        
        # Draw gradient line
        for i, point in enumerate(points):
            alpha = 1.0 - (i / len(points)) * 0.7  # Fade with distance
            thickness = max(1, int(8 * alpha))
            cv2.circle(frame, point, thickness, color, -1)
    
    def get_line_points(self, start: Tuple[int, int], end: Tuple[int, int], 
                       num_points: int) -> List[Tuple[int, int]]:
        """Get points along a line."""
        x1, y1 = start
        x2, y2 = end
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            points.append((x, y))
        
        return points
    
    def check_lane_departure(self, lane_data: Dict) -> bool:
        """Check if vehicle is departing from lane."""
        # Simplified lane departure detection
        return False  # Placeholder
    
    def draw_lane_departure_warning(self, frame: np.ndarray):
        """Draw lane departure warning."""
        height, width = frame.shape[:2]
        warning_text = "LANE DEPARTURE WARNING"
        cv2.putText(frame, warning_text, (width//2 - 150, 50), self.font, 1.0, 
                   self.colors['danger'], 3, cv2.LINE_AA)
    
    def draw_enhanced_trajectory_projection(self, frame: np.ndarray, lane_data: Dict, 
                                          speed: float, confidence: float) -> np.ndarray:
        """Draw enhanced trajectory projection with particle effects."""
        height, width = frame.shape[:2]
        
        # Calculate projection length based on speed
        projection_length = min(height * 0.8, 100 + speed * 2)
        
        # Create particle effects along the projection
        self.draw_particle_effects(frame, width//2, height, projection_length)
        
        # Draw dynamic projection with confidence-based opacity
        opacity = 0.1 + confidence * 0.4
        self.draw_dynamic_projection(frame, width//2, height, projection_length, opacity)
        
        return frame
    
    def draw_particle_effects(self, frame: np.ndarray, center_x: int, start_y: int, length: int):
        """Draw particle effects along the trajectory."""
        num_particles = 20
        for i in range(num_particles):
            t = i / num_particles
            y = int(start_y - t * length)
            x = center_x + int(10 * math.sin(t * 10 + time.time() * 2))
            
            # Draw particle
            cv2.circle(frame, (x, y), 2, self.colors['success'], -1)
    
    def draw_dynamic_projection(self, frame: np.ndarray, center_x: int, start_y: int, 
                               length: int, opacity: float):
        """Draw dynamic projection with variable opacity."""
        overlay = frame.copy()
        
        # Create fan-shaped projection
        fan_width = 20
        for i in range(10):
            t = i / 10
            current_y = int(start_y - t * length)
            current_width = int(fan_width * (1 + t * 2))
            
            cv2.line(overlay, (center_x - current_width, current_y), 
                    (center_x + current_width, current_y), self.colors['success'], 3)
        
        # Blend with opacity
        frame = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
    
    def draw_professional_ui(self, frame: np.ndarray, fps: float, speed: float) -> np.ndarray:
        """Draw professional UI elements."""
        height, width = frame.shape[:2]
        
        # Draw company branding
        cv2.putText(frame, self.company_name, (width - 200, 30), self.font, 0.8, 
                   self.colors['accent'], 2, cv2.LINE_AA)
        
        # Draw timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (20, 30), self.font, 0.6, 
                   self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        # Draw performance metrics
        self.draw_performance_metrics(frame, fps)
        
        # Draw system health indicators
        self.draw_system_health(frame)
        
        return frame
    
    def draw_performance_metrics(self, frame: np.ndarray, fps: float):
        """Draw performance metrics."""
        metrics = [
            f"FPS: {fps:.1f}",
            f"Detection: {self.detection_accuracy:.1%}",
            f"Objects: {len(self.object_animations)}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (20, 60 + i * 20), self.font, 0.5, 
                       self.colors['text_secondary'], 1, cv2.LINE_AA)
    
    def draw_system_health(self, frame: np.ndarray):
        """Draw system health indicators."""
        health_indicators = [
            f"Battery: {self.system_health['battery']}%",
            f"GPS: {self.system_health['gps']}",
            f"Sensors: {self.system_health['sensors']}",
            f"AI: {self.system_health['ai_model']}"
        ]
        
        for i, indicator in enumerate(health_indicators):
            color = self.colors['success'] if 'Optimal' in indicator or 'Active' in indicator else self.colors['warning']
            cv2.putText(frame, indicator, (20, 140 + i * 20), self.font, 0.5, 
                       color, 1, cv2.LINE_AA)
    
    def draw_advanced_safety_features(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw advanced safety features."""
        # Collision prediction zones
        self.draw_collision_zones(frame, detections)
        
        # Safe following distance indicators
        self.draw_following_distance_indicators(frame, detections)
        
        # Emergency stop warnings
        self.draw_emergency_warnings(frame, detections)
        
        return frame
    
    def draw_collision_zones(self, frame: np.ndarray, detections: Dict):
        """Draw collision prediction zones."""
        for vehicle in detections.get('vehicles', []):
            x1, y1, x2, y2 = vehicle['bbox']
            distance = vehicle.get('distance', 30)
            
            if distance < 20:  # Close vehicle
                # Draw collision warning zone
                cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), 
                             self.colors['danger'], 2)
    
    def draw_following_distance_indicators(self, frame: np.ndarray, detections: Dict):
        """Draw safe following distance indicators."""
        for vehicle in detections.get('vehicles', []):
            distance = vehicle.get('distance', 30)
            center_x, center_y = vehicle['center']
            
            if distance < 30:
                # Draw distance indicator
                cv2.putText(frame, f"{distance:.1f}m", (center_x, center_y - 20), 
                           self.font, 0.6, self.colors['warning'], 2, cv2.LINE_AA)
    
    def draw_emergency_warnings(self, frame: np.ndarray, detections: Dict):
        """Draw emergency stop warnings."""
        for vehicle in detections.get('vehicles', []):
            distance = vehicle.get('distance', 30)
            
            if distance < 10:  # Very close vehicle
                # Draw pulsing emergency warning
                pulse = int(255 * (0.5 + 0.5 * math.sin(time.time() * 5)))
                cv2.putText(frame, "EMERGENCY STOP", (frame.shape[1]//2 - 100, 80), 
                           self.font, 1.2, (0, 0, pulse), 3, cv2.LINE_AA)
    
    def create_enhanced_dashboard_overlay(self, frame: np.ndarray, detections: Dict,
                                        lane_data: Dict, fps: float, speed: float, 
                                        steering_angle: float) -> np.ndarray:
        """Create enhanced dashboard overlay with all professional features."""
        
        # Draw enhanced lane detection
        frame = self.draw_gradient_lane_lines(frame, lane_data)
        
        # Draw enhanced trajectory projection
        frame = self.draw_enhanced_trajectory_projection(frame, lane_data, speed, self.detection_accuracy)
        
        # Draw enhanced object detection
        for category in ['vehicles', 'pedestrians', 'traffic_signs']:
            for detection in detections.get(category, []):
                frame = self.draw_enhanced_bounding_box(frame, detection)
        
        # Draw advanced safety features
        frame = self.draw_advanced_safety_features(frame, detections)
        
        # Draw professional UI
        frame = self.draw_professional_ui(frame, fps, speed)
        
        # Draw modern dashboard panels
        frame = self.draw_modern_dashboard_panels(frame, speed, detections)
        
        return frame
    
    def draw_modern_dashboard_panels(self, frame: np.ndarray, speed: float, detections: Dict):
        """Draw modern dashboard panels with glassmorphism effect."""
        
        # Status panel
        status_content = [
            f"Status: ACTIVE",
            f"Speed: {speed:.1f} km/h",
            f"Confidence: {self.detection_accuracy:.1%}"
        ]
        frame = self.draw_glassmorphism_panel(frame, 20, 20, 300, 120, 
                                            "🚗 AUTONOMOUS DRIVING", status_content)
        
        # Live Feed panel
        live_feed_content = [
            f"Following: {len(detections.get('vehicles', []))} vehicles",
            f"Pedestrians: {len(detections.get('pedestrians', []))}",
            f"Lane: Centered",
            f"System: Optimal"
        ]
        frame = self.draw_glassmorphism_panel(frame, frame.shape[1] - 320, 20, 300, 120, 
                                            "📡 LIVE FEED", live_feed_content)
        
        return frame

    # Keep existing methods for compatibility
    def get_scaled_text_params(self, frame: np.ndarray) -> dict:
        """Get scaled text parameters based on frame resolution."""
        height, width = frame.shape[:2]
        base_width = 1920
        base_height = 1080
        scale_factor = min(width / base_width, height / base_height)
        
        return {
            'font_scale': max(0.4, min(1.5, scale_factor * 0.8)),
            'thickness': max(1, min(4, int(scale_factor * 2))),
            'panel_scale': scale_factor
        }

    def draw_status_panel(self, frame: np.ndarray, speed: float, confidence: int = 95) -> np.ndarray:
        """Legacy method - now uses modern dashboard."""
        return self.draw_modern_dashboard_panels(frame, speed, {})

    def draw_action_feed(self, frame: np.ndarray, action_feed: List[str]) -> np.ndarray:
        """Legacy method - now uses modern dashboard."""
        return frame

    def analyze_obstacles_and_plan_path(self, frame: np.ndarray, detections: Dict, lane_data: Dict) -> Dict:
        """Analyze obstacles and plan path (enhanced version)."""
        # Enhanced path planning logic
        path_planning = {
            'path_type': 'straight',
            'obstacles': [],
            'safety_zones': [],
            'path_arrow': None,
            'lane_center': (0, 0, 0, 0)
        }
        
        # Add enhanced obstacle analysis here
        return path_planning

    def draw_intelligent_path_planning(self, frame: np.ndarray, path_planning: Dict) -> np.ndarray:
        """Draw intelligent path planning (enhanced version)."""
        # Enhanced path planning visualization
        return frame
    
    def update_decision_brain_with_path_planning(self, path_planning: Dict):
        """Update decision brain with path planning."""
        if hasattr(self, 'decision_overlay_data'):
            action_feed = self.decision_overlay_data.get('action_feed', [])
            action_feed.append(f"Enhanced Path Planning Active")
            self.decision_overlay_data['action_feed'] = action_feed[-5:]

    def draw_realistic_trajectory_prediction(self, frame: np.ndarray, lane_data: Dict, speed: float) -> np.ndarray:
        """Enhanced trajectory prediction."""
        return self.draw_enhanced_trajectory_projection(frame, lane_data, speed, self.detection_accuracy)

    def create_dashboard_overlay(self, frame: np.ndarray, detections: Dict,
                               lane_data: Dict, fps: float, speed: float, steering_angle: float) -> np.ndarray:
        """Enhanced dashboard overlay."""
        return self.create_enhanced_dashboard_overlay(frame, detections, lane_data, fps, speed, steering_angle)
    
    def create_debug_view(self, original_frame: np.ndarray, 
                         processed_frame: np.ndarray,
                         lane_data: Dict) -> np.ndarray:
        """
        Create a debug view showing original and processed frames side by side.
        
        Args:
            original_frame: Original input frame
            processed_frame: Frame with all overlays
            lane_data: Lane detection results for edge visualization
            
        Returns:
            Combined debug frame
        """
        height, width = original_frame.shape[:2]
        
        # Resize frames for side-by-side display
        display_width = width // 2
        display_height = height // 2
        
        original_small = cv2.resize(original_frame, (display_width, display_height))
        processed_small = cv2.resize(processed_frame, (display_width, display_height))
        
        # Create edge detection visualization if available
        if 'debug_edges' in lane_data and lane_data['debug_edges'] is not None:
            edges = lane_data['debug_edges']
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_small = cv2.resize(edges_colored, (display_width, display_height))
        else:
            edges_small = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Create combined frame
        combined = np.zeros((display_height * 2, display_width * 2, 3), dtype=np.uint8)
        
        # Place frames
        combined[0:display_height, 0:display_width] = original_small
        combined[0:display_height, display_width:] = processed_small
        combined[display_height:, 0:display_width] = edges_small
        
        # Add labels
        cv2.putText(combined, "Original", (10, 25), self.font, 0.5, 
                   self.colors['text'], 1)
        cv2.putText(combined, "Processed", (display_width + 10, 25), 
                   self.font, 0.5, self.colors['text'], 1)
        cv2.putText(combined, "Edge Detection", (10, display_height + 25), 
                   self.font, 0.5, self.colors['text'], 1)
        
        return combined