import cv2
import numpy as np

class Visualizer:
    """
    Handles all OSD (On-Screen Display) rendering.
    """
    def __init__(self):
        # Colors (B, G, R)
        self.COLOR_HAND = (0, 255, 0)      # Green skeleton
        self.COLOR_TEXT = (255, 255, 255)  # White text
        self.COLOR_BG = (0, 0, 0)          # Black background box
        
        # Connections for wireframe drawing
        self.CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]

    def draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draws the 3D skeleton projected onto the 2D frame.
        Landmarks should be denormalized to pixel coordinates before calling this.
        """
        if landmarks is None:
            return frame

        h, w, _ = frame.shape
        
        # Convert floats to ints
        # Assumes landmarks are (21, 3) or (21, 2)
        points = []
        for point in landmarks:
            # Denormalize if normalized [0,1], otherwise assume pixel coords
            px, py = int(point[0]), int(point[1])
            points.append((px, py))
            cv2.circle(frame, (px, py), 4, self.COLOR_HAND, -1)

        # Draw bones
        for start_idx, end_idx in self.CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], self.COLOR_HAND, 2)
            
        return frame

    def draw_overlay(self, frame: np.ndarray, equation: str, result: str, prediction_label: str = "") -> np.ndarray:
        """
        Draws the HUD with equation and result.
        """
        h, w, _ = frame.shape
        
        # 1. Top Bar (Equation)
        if equation:
            cv2.rectangle(frame, (0, 0), (w, 60), self.COLOR_BG, -1)
            cv2.putText(frame, f"Expr: {equation}", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOR_TEXT, 2)

        # 2. Bottom Bar (Result)
        if result:
            cv2.rectangle(frame, (0, h-80), (w, h), (0, 100, 0), -1) # Dark Green for Success
            cv2.putText(frame, f"RESULT: {result}", (20, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_TEXT, 3)
            
        # 3. Live Prediction (Debug)
        if prediction_label:
            cv2.putText(frame, f"Live: {prediction_label}", (w - 200, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            
        return frame