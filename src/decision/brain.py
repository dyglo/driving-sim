"""
Decision-making module for autonomous driving simulation.
Implements behavioral intelligence, risk assessment, and decision feedback.
"""

from typing import Dict, List
import time

SAFE_FOLLOWING_DISTANCE = 20  # meters

class AutonomousDrivingBrain:
    def __init__(self):
        self.last_decisions = []
        self.last_action_feed = []
        self.confidence = 95  # Simulated confidence value
        self.last_update_time = time.time()

    def analyze_and_decide(self, detected_objects: Dict, lane_info: Dict, speed: float, steering_angle: float) -> Dict:
        decisions = []
        action_feed = []
        now = time.strftime('%H:%M:%S')
        # Vehicle following logic
        for vehicle in detected_objects.get('vehicles', []):
            distance = vehicle.get('distance', 30)
            if distance < SAFE_FOLLOWING_DISTANCE:
                decisions.append("REDUCING SPEED: MAINTAINING SAFE DISTANCE")
                action_feed.append(f"{now} FOLLOW: {distance:.1f}m - REDUCING SPEED")
            else:
                action_feed.append(f"{now} FOLLOW: {distance:.1f}m - SAFE")
        # Lane management
        if lane_info.get('departure', False):
            decisions.append("MINOR LEFT CORRECTION FOR LANE CENTER")
            action_feed.append(f"{now} LANE: Lane Departure Detected")
        else:
            action_feed.append(f"{now} LANE: Centered")
        # Speed management
        if speed > 80:
            decisions.append("REDUCING SPEED: TRAFFIC AHEAD SLOWING")
            action_feed.append(f"{now} SPEED: {speed:.1f} km/h - SLOWING")
        else:
            action_feed.append(f"{now} SPEED: {speed:.1f} km/h - OK")
        # Steering
        if abs(steering_angle) > 10:
            decisions.append("STEERING: MAJOR CORRECTION")
            action_feed.append(f"{now} STEER: {steering_angle:.1f} deg - CORRECTING")
        else:
            action_feed.append(f"{now} STEER: {steering_angle:.1f} deg - STABLE")
        # System status
        decisions.append("SYSTEM STATUS: AUTONOMOUS DRIVING ACTIVE")
        # Keep last 5 actions
        self.last_action_feed = (action_feed + self.last_action_feed)[0:5]
        self.last_decisions = decisions
        return {
            'decisions': decisions,
            'action_feed': self.last_action_feed,
            'confidence': self.confidence
        }
