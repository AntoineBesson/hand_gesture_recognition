import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import sys

# Configuration
DATA_ROOT = "data/raw"
MODEL_PATH = "hand_landmarker.task"  # <--- DOIT ÊTRE À LA RACINE DU PROJET
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus", "minus", "mult", "div", "equal"]
WINDOW_SIZE = 64

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Modèle manquant '{MODEL_PATH}'")
        print("Téléchargez-le ici: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)

def ensure_dirs():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
    for cls in CLASSES:
        os.makedirs(os.path.join(DATA_ROOT, cls), exist_ok=True)

# --- VISUALIZATION ENGINE (Custom implementation to avoid mp.solutions) ---
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    # Connections (MediaPipe Topology)
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    COLOR_DOT = (0, 255, 0)
    COLOR_LINE = (255, 255, 255)

    for hand_landmarks in hand_landmarks_list:
        # Convert normalized coords to pixels
        px_points = []
        for lm in hand_landmarks:
            px_points.append((int(lm.x * w), int(lm.y * h)))

        # Draw Lines
        for start_idx, end_idx in CONNECTIONS:
            cv2.line(annotated_image, px_points[start_idx], px_points[end_idx], COLOR_LINE, 2)
            
        # Draw Points
        for x, y in px_points:
            cv2.circle(annotated_image, (x, y), 4, COLOR_DOT, -1)

    return annotated_image

def record_sample(class_name, cap, landmarker):
    print(f"--- RECORDING {class_name} ---")
    sequence_data = []
    
    start_time_ms = int(time.time() * 1000)
    
    while len(sequence_data) < WINDOW_SIZE:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Flip & Format
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 3. Process (VIDEO MODE REQUIREMENT: Monotonic Timestamp)
        # We synthesize a timestamp based on capture time
        frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # 4. Extract Data (Dual Hand Logic)
        frame_hands = np.zeros((2, 21, 3)) 
        detected_labels = []
        
        # The new API result structure:
        # detection_result.hand_landmarks: List[List[NormalizedLandmark]]
        # detection_result.handedness: List[List[Category]]
        
        if detection_result.hand_landmarks:
            for idx, hand_lms in enumerate(detection_result.hand_landmarks):
                # Get Handedness (Left/Right)
                # Note: Tasks API returns 'Left'/'Right' lists corresponding to landmarks
                handedness_category = detection_result.handedness[idx][0]
                label = handedness_category.category_name # 'Left' or 'Right'
                
                # Convert Normalized [0,1] back to Pixel Space [w, h]
                # We do this to maintain compatibility with your training pipeline logic
                xyz = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lms])
                
                if label == 'Left':
                    frame_hands[0] = xyz
                    detected_labels.append("L")
                else:
                    frame_hands[1] = xyz
                    detected_labels.append("R")
        
        sequence_data.append(frame_hands)
        
        # 5. Visualization (Custom)
        annotated_frame = draw_landmarks_on_image(frame, detection_result)
        
        # UI Feedback
        cv2.rectangle(annotated_frame, (0,0), (w,h), (0,0,255), 6)
        status = f"REC: {class_name} | {len(sequence_data)}/{WINDOW_SIZE}"
        cv2.putText(annotated_frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        hand_status = f"Hands: {'+'.join(detected_labels) if detected_labels else 'NONE'}"
        color = (0, 255, 0) if detected_labels else (0, 0, 255)
        cv2.putText(annotated_frame, hand_status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('NeuroCalc Recorder', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
            
    # Save
    sequence_data = np.array(sequence_data)
    timestamp = int(time.time() * 1000)
    filename = os.path.join(DATA_ROOT, class_name, f"seq_{timestamp}.npy")
    np.save(filename, sequence_data)
    print(f"Saved: {filename}")
    return True

def main():
    ensure_model_exists()
    ensure_dirs()
    
    # --- NEW API INITIALIZATION ---
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Context Manager for automatic cleanup
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        
        # Initialize Sensor (Try indexes 0, 1, 2 if 0 fails)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             cap = cv2.VideoCapture(1)
        
        current_idx = 0
        
        print("\n=== NEURO_CALC RECORDER (TASKS API) ===")
        print(f"Model: {MODEL_PATH}")
        print("Controls: [SPACE] Record, [n] Next, [p] Prev, [q] Quit")
        
        start_time_ms = int(time.time() * 1000)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed.")
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Passive Tracking for UI
            frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            # Draw
            vis_frame = draw_landmarks_on_image(frame, detection_result)
            
            # UI Overlay
            target_cls = CLASSES[current_idx]
            count = len(os.listdir(os.path.join(DATA_ROOT, target_cls)))
            
            cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], 80), (50, 50, 50), -1)
            cv2.putText(vis_frame, f"TARGET: {target_cls.upper()} ({count})", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('NeuroCalc Recorder', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_idx = (current_idx + 1) % len(CLASSES)
            elif key == ord('p'):
                current_idx = (current_idx - 1) % len(CLASSES)
            elif key == 32: # SPACE
                success = record_sample(target_cls, cap, landmarker)
                # Reset base time after heavy recording operation to ensure monotonic consistency
                start_time_ms = int(time.time() * 1000)
                if not success: break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()