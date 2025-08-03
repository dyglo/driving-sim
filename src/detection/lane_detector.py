"""
OpenCV-based lane detection for autonomous driving simulation.
Uses Canny edge detection and Hough line transformation to detect road lanes.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from ..utils.config import LANE_DETECTION

class LaneDetector:
    """
    Lane detection using traditional computer vision techniques.
    """
    
    def __init__(self):
        """Initialize the lane detector with configuration parameters."""
        self.config = LANE_DETECTION
        self.prev_left_lane = None
        self.prev_right_lane = None
        
    def create_roi_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a region of interest mask focusing on the road area.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (mask, roi_vertices)
        """
        height, width = frame.shape[:2]
        
        # Define ROI as a trapezoid focusing on the road
        roi_height = int(height * 0.6)  # Focus on bottom 60% of image
        roi_vertices = np.array([
            [0, height],                                    # Bottom left
            [width * 0.45, roi_height],                    # Top left
            [width * 0.55, roi_height],                    # Top right
            [width, height]                                 # Bottom right
        ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_vertices], 255)
        
        return mask, roi_vertices
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for lane detection.
        
        Args:
            frame: Input color frame
            
        Returns:
            Preprocessed binary edge image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray, 
            (5, 5), 
            0
        )
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.config["canny_low"],
            self.config["canny_high"]
        )
        
        return edges
    
    def detect_lines(self, edges: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """
        Detect lines using Hough line transformation.
        
        Args:
            edges: Binary edge image
            mask: ROI mask
            
        Returns:
            List of detected lines
        """
        # Apply ROI mask
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=self.config["hough_rho"],
            theta=self.config["hough_theta"],
            threshold=self.config["hough_threshold"],
            minLineLength=self.config["hough_min_line_length"],
            maxLineGap=self.config["hough_max_line_gap"]
        )
        
        return lines if lines is not None else []
    
    def classify_lines(self, lines: List[np.ndarray], 
                      frame_width: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Classify lines as left or right lane based on slope and position.
        
        Args:
            lines: List of detected lines
            frame_width: Width of the frame
            
        Returns:
            Tuple of (left_lines, right_lines)
        """
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:  # Vertical line
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter lines based on slope and position
            line_center_x = (x1 + x2) / 2
            
            # Left lane: negative slope, left side of frame
            if slope < -0.3 and line_center_x < frame_width * 0.6:
                left_lines.append(line)
            # Right lane: positive slope, right side of frame
            elif slope > 0.3 and line_center_x > frame_width * 0.4:
                right_lines.append(line)
        
        return left_lines, right_lines
    
    def fit_lane_line(self, lines: List[np.ndarray], 
                     frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Fit a single lane line from multiple line segments.
        
        Args:
            lines: List of line segments for one lane
            frame_height: Height of the frame
            
        Returns:
            Single lane line coordinates (x1, y1, x2, y2) or None
        """
        if not lines:
            return None
        
        # Extract all points from lines
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
        
        if len(points) < 2:
            return None
        
        # Fit line using least squares
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        try:
            # Fit polynomial (degree 1 for straight line)
            coeffs = np.polyfit(y_coords, x_coords, 1)
            
            # Calculate line endpoints
            y1 = frame_height  # Bottom of frame
            y2 = int(frame_height * 0.6)  # Top of ROI
            
            x1 = int(coeffs[0] * y1 + coeffs[1])
            x2 = int(coeffs[0] * y2 + coeffs[1])
            
            return (x1, y1, x2, y2)
            
        except np.linalg.LinAlgError:
            return None
    
    def smooth_lanes(self, left_lane: Optional[Tuple], 
                    right_lane: Optional[Tuple]) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Smooth lane detection using previous frame information.
        
        Args:
            left_lane: Current left lane line
            right_lane: Current right lane line
            
        Returns:
            Smoothed lane lines
        """
        alpha = 0.3  # Smoothing factor
        
        # Smooth left lane
        if left_lane is not None:
            if self.prev_left_lane is not None:
                # Weighted average with previous detection
                smoothed_left = tuple(
                    int(alpha * curr + (1 - alpha) * prev)
                    for curr, prev in zip(left_lane, self.prev_left_lane)
                )
                self.prev_left_lane = smoothed_left
            else:
                self.prev_left_lane = left_lane
                smoothed_left = left_lane
        else:
            smoothed_left = self.prev_left_lane
        
        # Smooth right lane
        if right_lane is not None:
            if self.prev_right_lane is not None:
                # Weighted average with previous detection
                smoothed_right = tuple(
                    int(alpha * curr + (1 - alpha) * prev)
                    for curr, prev in zip(right_lane, self.prev_right_lane)
                )
                self.prev_right_lane = smoothed_right
            else:
                self.prev_right_lane = right_lane
                smoothed_right = right_lane
        else:
            smoothed_right = self.prev_right_lane
        
        return smoothed_left, smoothed_right
    
    def detect_lanes(self, frame: np.ndarray) -> dict:
        """
        Main method to detect lanes in a frame.
        
        Args:
            frame: Input color frame
            
        Returns:
            Dictionary containing lane detection results
        """
        height, width = frame.shape[:2]
        
        try:
            # Create ROI mask
            mask, roi_vertices = self.create_roi_mask(frame)
            
            # Preprocess frame
            edges = self.preprocess_frame(frame)
            
            # Detect lines
            lines = self.detect_lines(edges, mask)
            
            if lines is None or len(lines) == 0:
                return {
                    'left_lane': None,
                    'right_lane': None,
                    'roi_vertices': roi_vertices,
                    'lane_region': None,
                    'status': 'no_lines_detected'
                }
            
            # Classify lines as left or right
            left_lines, right_lines = self.classify_lines(lines, width)
            
            # Fit lane lines
            left_lane = self.fit_lane_line(left_lines, height)
            right_lane = self.fit_lane_line(right_lines, height)
            
            # Apply smoothing
            left_lane, right_lane = self.smooth_lanes(left_lane, right_lane)
            
            # Create lane region polygon if both lanes detected
            lane_region = None
            if left_lane and right_lane:
                x1_l, y1_l, x2_l, y2_l = left_lane
                x1_r, y1_r, x2_r, y2_r = right_lane
                
                lane_region = np.array([
                    [x1_l, y1_l],  # Bottom left
                    [x2_l, y2_l],  # Top left
                    [x2_r, y2_r],  # Top right
                    [x1_r, y1_r]   # Bottom right
                ], dtype=np.int32)
            
            return {
                'left_lane': left_lane,
                'right_lane': right_lane,
                'roi_vertices': roi_vertices,
                'lane_region': lane_region,
                'status': 'success',
                'debug_edges': edges  # For debugging
            }
            
        except Exception as e:
            logging.error(f"Lane detection error: {e}")
            return {
                'left_lane': None,
                'right_lane': None,
                'roi_vertices': np.array([]),
                'lane_region': None,
                'status': f'error: {str(e)}'
            }
    
    def get_lane_curvature(self, left_lane: Tuple, right_lane: Tuple) -> str:
        """
        Estimate lane curvature direction.
        
        Args:
            left_lane: Left lane line coordinates
            right_lane: Right lane line coordinates
            
        Returns:
            Curvature direction string
        """
        if left_lane is None or right_lane is None:
            return "unknown"
        
        # Calculate midpoint of lanes at different heights
        x1_mid = (left_lane[0] + right_lane[0]) / 2  # Bottom midpoint
        x2_mid = (left_lane[2] + right_lane[2]) / 2  # Top midpoint
        
        # Determine curve direction
        curve_diff = x2_mid - x1_mid
        
        if abs(curve_diff) < 10:
            return "straight"
        elif curve_diff > 0:
            return "right_curve"
        else:
            return "left_curve"