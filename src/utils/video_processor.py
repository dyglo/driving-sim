"""
Video processing utilities for the autonomous driving simulation.
Handles video input/output, frame processing, and real-time display.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Callable, Tuple
from pathlib import Path

from .config import VIDEO_CONFIG, PERFORMANCE_CONFIG

class VideoProcessor:
    """
    Handles video input/output and frame processing for the simulation.
    """
    
    def __init__(self, input_path: str, output_path: Optional[str] = None):
        """
        Initialize the video processor.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file (optional)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else None
        
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps_original = 0
        self.fps_current = 0
        self.frame_width = 0
        self.frame_height = 0
        
        # Performance tracking
        self.process_times = []
        self.frame_skip_counter = 0
        self.current_frame_idx = 0
        
    def open_video(self) -> bool:
        """
        Open the input video file and initialize properties.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(str(self.input_path))
            
            if not self.cap.isOpened():
                logging.error(f"Could not open video file: {self.input_path}")
                return False
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps_original = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logging.info(f"Video opened: {self.frame_width}x{self.frame_height} @ {self.fps_original} FPS")
            logging.info(f"Total frames: {self.total_frames}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error opening video: {e}")
            return False
    
    def setup_output_writer(self) -> bool:
        """
        Setup video writer for output file.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.output_path:
            return True  # No output file requested
        
        try:
            # Create output directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Define codec
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG['codec'])
            
            # Use original FPS or configured FPS
            output_fps = min(self.fps_original, VIDEO_CONFIG['fps'])
            
            # Initialize video writer
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                output_fps,
                (self.frame_width, self.frame_height)
            )
            
            if not self.writer.isOpened():
                logging.error(f"Could not open output video writer: {self.output_path}")
                return False
            
            logging.info(f"Output video writer initialized: {self.output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error setting up output writer: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.cap:
            return False, None
        
        # Skip frames if configured for performance
        for _ in range(PERFORMANCE_CONFIG.get('skip_frames', 1) - 1):
            ret, _ = self.cap.read()
            if not ret:
                return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            
            # Resize frame if configured
            if PERFORMANCE_CONFIG.get('resize_factor', 1.0) != 1.0:
                new_width = int(self.frame_width * PERFORMANCE_CONFIG.get('resize_factor', 1.0))
                new_height = int(self.frame_height * PERFORMANCE_CONFIG.get('resize_factor', 1.0))
                frame = cv2.resize(frame, (new_width, new_height))
        
        return ret, frame
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if successful, False otherwise
        """
        if not self.writer:
            return True  # No output writer configured
        
        try:
            # Ensure frame is the correct size
            if frame.shape[:2] != (self.frame_height, self.frame_width):
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            self.writer.write(frame)
            return True
            
        except Exception as e:
            logging.error(f"Error writing frame: {e}")
            return False
    
    def display_frame(self, frame: np.ndarray, window_name: str = "Autonomous Driving Simulation") -> bool:
        """
        Display frame in a window.
        
        Args:
            frame: Frame to display
            window_name: Name of the display window
            
        Returns:
            True to continue, False to stop (user pressed 'q')
        """
        if not VIDEO_CONFIG['show_realtime']:
            return True
        
        try:
            # Resize for display if needed
            display_frame = frame
            if (frame.shape[1] != VIDEO_CONFIG['display_width'] or 
                frame.shape[0] != VIDEO_CONFIG['display_height']):
                display_frame = cv2.resize(
                    frame, 
                    (VIDEO_CONFIG['display_width'], VIDEO_CONFIG['display_height'])
                )
            
            cv2.imshow(window_name, display_frame)
            
            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logging.info("User requested stop")
                return False
            elif key == ord('p'):
                # Pause functionality
                logging.info("Paused - press any key to continue")
                cv2.waitKey(0)
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                save_path = f"frame_{timestamp}.jpg"
                cv2.imwrite(save_path, frame)
                logging.info(f"Frame saved as {save_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error displaying frame: {e}")
            return True
    
    def calculate_fps(self) -> float:
        """
        Calculate current processing FPS.
        
        Returns:
            Current FPS
        """
        if len(self.process_times) < 2:
            return 0.0
        
        # Calculate FPS from recent processing times
        recent_times = self.process_times[-10:]  # Last 10 frames
        if len(recent_times) >= 2:
            time_diff = recent_times[-1] - recent_times[0]
            fps = (len(recent_times) - 1) / time_diff if time_diff > 0 else 0
            return fps
        
        return 0.0
    
    def update_performance_stats(self, process_start_time: float):
        """
        Update performance statistics.
        
        Args:
            process_start_time: Start time of frame processing
        """
        current_time = time.time()
        self.process_times.append(current_time)
        
        # Keep only recent times for FPS calculation
        if len(self.process_times) > 30:
            self.process_times = self.process_times[-20:]
        
        self.fps_current = self.calculate_fps()
    
    def get_progress(self) -> dict:
        """
        Get processing progress information.
        
        Returns:
            Dictionary with progress information
        """
        progress_percent = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
        
        return {
            'current_frame': self.frame_count,
            'total_frames': self.total_frames,
            'progress_percent': progress_percent,
            'fps_original': self.fps_original,
            'fps_current': self.fps_current,
            'estimated_time_remaining': self._estimate_time_remaining()
        }
    
    def _estimate_time_remaining(self) -> float:
        """
        Estimate remaining processing time.
        
        Returns:
            Estimated time in seconds
        """
        if self.fps_current <= 0 or self.total_frames <= 0:
            return 0.0
        
        remaining_frames = self.total_frames - self.frame_count
        return remaining_frames / self.fps_current
    
    def process_video(self, frame_processor: Callable[[np.ndarray], np.ndarray],
                     progress_callback: Optional[Callable[[dict], None]] = None) -> bool:
        """
        Process entire video with given frame processor function.
        
        Args:
            frame_processor: Function to process each frame
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        if not self.open_video():
            return False
        
        if VIDEO_CONFIG['save_output']:
            if not self.setup_output_writer():
                return False
        
        logging.info("Starting video processing...")
        start_time = time.time()
        
        try:
            while True:
                # Update current frame index
                if self.cap:
                    self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                process_start = time.time()
                
                # Read frame
                ret, frame = self.read_frame()
                if not ret:
                    break
                
                # Process frame
                processed_frame = frame_processor(frame.copy())
                
                # Write to output if configured
                if VIDEO_CONFIG['save_output']:
                    self.write_frame(processed_frame)
                
                # Display frame
                if not self.display_frame(processed_frame):
                    break  # User requested stop
                
                # Update performance stats
                self.update_performance_stats(process_start)
                
                # Progress callback
                if progress_callback and self.frame_count % 30 == 0:  # Every 30 frames
                    progress_callback(self.get_progress())
                
                # Log progress periodically
                if self.frame_count % 100 == 0:
                    progress = self.get_progress()
                    logging.info(
                        f"Progress: {progress['progress_percent']:.1f}% "
                        f"({progress['current_frame']}/{progress['total_frames']}) "
                        f"FPS: {progress['fps_current']:.1f}"
                    )
            
            # Processing complete
            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            logging.info(f"Video processing complete!")
            logging.info(f"Processed {self.frame_count} frames in {total_time:.2f} seconds")
            logging.info(f"Average FPS: {avg_fps:.2f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error during video processing: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        cv2.destroyAllWindows()
        logging.info("Video processor cleanup complete")
    
    def get_video_info(self) -> dict:
        """
        Get detailed video information.
        
        Returns:
            Dictionary with video information
        """
        if not self.cap:
            return {}
        
        return {
            'file_path': str(self.input_path),
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.fps_original,
            'total_frames': self.total_frames,
            'duration_seconds': self.total_frames / self.fps_original if self.fps_original > 0 else 0,
            'codec': self.cap.get(cv2.CAP_PROP_FOURCC)
        }