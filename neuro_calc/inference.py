import cv2
import torch
import hydra
import mediapipe as mp
import numpy as np
from collections import deque
from omegaconf import DictConfig

# Import Stack
from src.models.st_gcn import HandSignRecognizer
from src.core.geometry import process_dual_hand_frame # <--- The new dual-hand function
from src.core.solver import GestureSolver
from src.pipeline.visualizer import Visualizer

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Device Setup
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Initializing Dual-Hand Inference Engine on {device}...")

    # 2. Load Model
    # Note: Ensure your config.yaml model definition expects 42 vertices now!
    model = HandSignRecognizer(num_classes=len(cfg.classes))
    
    checkpoint_path = "best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL WARNING: {checkpoint_path} not found. Running with random weights.")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()

    # 3. Initialize Components
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,               # <--- ENABLING DUAL HANDS
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    solver = GestureSolver(stability_frames=8) 
    viz = Visualizer()
    
    # 4. Temporal Buffer
    # Stores T frames of (42, 3) data (Unified Scene)
    window_size = cfg.model.window_size 
    frame_buffer = deque(maxlen=window_size)
    
    # 5. Sensor Loop
    cap = cv2.VideoCapture(0)
    
    print("System Online. Hands required: 1 or 2. Press 'q' to exit.")
    
    with torch.no_grad():
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # Flip for mirror effect (Natural interaction)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb_frame)
            
            # Containers for this frame's raw data
            left_hand_raw = None
            right_hand_raw = None
            
            # --- PARSING & VISUALIZATION ---
            if results.multi_hand_landmarks:
                for idx, handedness in enumerate(results.multi_handedness):
                    # Label is 'Left' or 'Right'
                    label = handedness.classification[0].label
                    lm = results.multi_hand_landmarks[idx]
                    
                    # Convert to pixel coords for processing
                    # (21, 3) Array
                    xyz = np.array([[l.x * w, l.y * h, l.z * w] for l in lm.landmark])
                    
                    if label == 'Left':
                        left_hand_raw = xyz
                    else:
                        right_hand_raw = xyz
                        
                    # Visualization (Draw Raw MediaPipe output)
                    # We draw each hand individually as they appear on screen
                    frame = viz.draw_skeleton(frame, xyz) 
            
            # --- PREPROCESSING (GEOMETRIC KERNEL) ---
            # Even if hands are None, the processor handles zero-padding
            # Returns: (42, 3) Tensor centered at scene midpoint
            dual_hand_canonical = process_dual_hand_frame(left_hand_raw, right_hand_raw)
            
            # Push to buffer
            frame_buffer.append(dual_hand_canonical)

            # --- INFERENCE TRIGGER ---
            equation, result = "", ""
            pred_label = ""
            
            if len(frame_buffer) == window_size:
                # Prepare Tensor: (1, 3, T, 42)
                input_tensor = np.array(frame_buffer) # (T, 42, 3)
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
                
                # Permute to PyTorch Format: (Batch, Channels, Time, Vertices)
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) 
                input_tensor = input_tensor.to(device)
                
                # Forward Pass
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                
                # Get Prediction
                conf, idx = torch.max(probs, 1)
                idx = idx.item()
                conf = conf.item()
                
                # Solve Logic
                pred_label = cfg.classes[idx]
                equation, result = solver.process_frame(idx, conf)
            
            # --- HUD RENDER ---
            status_text = f"Live: {pred_label}" if len(frame_buffer)==window_size else f"Buffering {len(frame_buffer)}/{window_size}..."
            frame = viz.draw_overlay(frame, equation, result, status_text)
            
            cv2.imshow('NeuroCalc Dual-Vision', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()