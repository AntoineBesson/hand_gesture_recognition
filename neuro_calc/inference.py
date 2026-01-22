import cv2
import torch
import hydra
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
from omegaconf import DictConfig
import time
import os

# Import Stack
from src.models.st_gcn import HandSignRecognizer
from src.core.geometry import process_dual_hand_frame
from src.core.solver import GestureSolver

# --- VISUALIZATION HELPER (Zero-Dependency) ---
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    # Topology lines
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    for hand_landmarks in hand_landmarks_list:
        px_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in CONNECTIONS:
            cv2.line(annotated_image, px_points[start], px_points[end], (200, 200, 200), 2)
        for x, y in px_points:
            cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), -1)
            
    return annotated_image

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Device Setup
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"--- NEURO_CALC INFERENCE ENGINE ({device}) ---")

    # 2. Load Model (Dual Hand Configuration)
    model = HandSignRecognizer(
        num_classes=len(cfg.classes),
    )
    
    checkpoint_path = "best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("ERROR: best_model.pth not found! Train the model first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 3. Initialize Vision (Tasks API)
    model_asset_path = "hand_landmarker.task" # Ensure this file is in root!
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    solver = GestureSolver(stability_frames=8)
    
    # 4. Temporal Buffer (Unified Scene)
    window_size = cfg.model.window_size 
    frame_buffer = deque(maxlen=window_size)
    
    # 5. Sensor Loop
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1) # Try 1 if 0 fails
    
    print("System Online. Calculating...")
    
    start_time_ms = int(time.time() * 1000)
    INFERENCE_STRIDE = 4 
    frame_count = 0

    # Cached results for smooth rendering
    last_pred_label = ""
    last_conf = 0.0
    last_equation = ""
    last_result = ""

    with torch.no_grad(), vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Tracking
            frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            frame_count += 1

            # --- EXTRACT HANDS ---
            left_hand_raw = None
            right_hand_raw = None
            
            if detection_result.hand_landmarks:
                for idx, hand_lms in enumerate(detection_result.hand_landmarks):
                    # Get Label (Left/Right)
                    label = detection_result.handedness[idx][0].category_name
                    
                    # Convert to Pixel Space (Robust Inputs)
                    xyz = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lms])
                    
                    if label == 'Left': left_hand_raw = xyz
                    else: right_hand_raw = xyz
            
            # --- GEOMETRIC KERNEL ---
            # Process scene even if hands are missing (returns zero-padded topology)
            dual_hand_canonical = process_dual_hand_frame(left_hand_raw, right_hand_raw)
            frame_buffer.append(dual_hand_canonical)

            # --- INFERENCE ---
            equation, result = "", ""
            pred_label = ""
            conf = 0.0
            
            if len(frame_buffer) == window_size and (frame_count % INFERENCE_STRIDE == 0):
                # Prepare Tensor: (1, 3, T, 42)
                input_tensor = np.array(frame_buffer)
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) 
                input_tensor = input_tensor.to(device)
                
                # Forward Pass
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                
                conf, idx = torch.max(probs, 1)
                idx = idx.item()
                conf = conf.item()
                
                # Update cached values
                last_pred_label = cfg.classes[idx]
                last_conf = conf
                last_equation, last_result = solver.process_frame(idx, conf)
            
            # --- RENDER (Always run at full FPS using cached values) ---
            vis_frame = draw_landmarks_on_image(frame, detection_result)
            
            # HUD - Top: Equation
            if last_equation:
                cv2.rectangle(vis_frame, (0, 0), (w, 60), (0,0,0), -1)
                cv2.putText(vis_frame, f"EQ: {last_equation}", (20, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Bottom: Result
            if last_result:
                cv2.rectangle(vis_frame, (0, h-80), (w, h), (0, 100, 0), -1)
                cv2.putText(vis_frame, f"= {last_result}", (20, h-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Debug Overlay (Top Right)
            color = (0, 255, 0) if last_conf > 0.8 else (0, 0, 255)
            cv2.putText(vis_frame, f"Pred: {last_pred_label} ({last_conf:.2f})", (w-250, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('NeuroCalc Live', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()