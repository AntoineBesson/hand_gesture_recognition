import numpy as np

def compute_basis_and_project(landmarks: np.ndarray) -> np.ndarray:
    """
    Performs rigid body transformation to align hand landmarks to the canonical frame.
    
    Args:
        landmarks: np.ndarray shape (21, 3) representing (x, y, z)
    
    Returns:
        np.ndarray: shape (21, 3) invariant to global rotation/translation.
    """
    # 0. Topology Constants (MediaPipe standard)
    WRIST_IDX = 0
    MIDDLE_MCP_IDX = 9
    PINKY_MCP_IDX = 17

    # 1. Translation: Center global frame at Wrist
    # T(p) = p - p_wrist
    wrist_pos = landmarks[WRIST_IDX]
    centered_marks = landmarks - wrist_pos

    # 2. Rotation: Gram-Schmidt Process
    # Primary Axis (Y-axis): Wrist -> Middle Finger Base
    v_primary = centered_marks[MIDDLE_MCP_IDX]
    y_axis = v_primary / (np.linalg.norm(v_primary) + 1e-6)

    # Secondary Axis Helper: Wrist -> Pinky Base
    v_secondary = centered_marks[PINKY_MCP_IDX]
    
    # Normal Vector (Z-axis): Cross Product
    # The normal to the palm plane
    z_axis = np.cross(y_axis, v_secondary)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)

    # Orthogonal X-axis: Cross Product of Y and Z
    x_axis = np.cross(y_axis, z_axis) # Already normalized
    
    # 3. Construct Basis Matrix (Rotation Matrix R)
    # R = [x_axis, y_axis, z_axis]^T
    R = np.stack([x_axis, y_axis, z_axis]) # Shape (3, 3)

    # 4. Project points into new basis
    # P_local = R * P_global^T
    # We transpose R because we are operating on row vectors (N, 3)
    # Result = centered_marks @ R.T
    return np.dot(centered_marks, R.T)

def vectorize_sequence_canonicalization(sequence: np.ndarray) -> np.ndarray:
    """
    Applies the canonical projection to an entire temporal sequence.
    Input: (T, 21, 3)
    Output: (T, 21, 3)
    """
    # For a production pipeline, we would rewrite the Gram-Schmidt 
    # using einsum for batch processing. For now, a list comprehension 
    # wrapped in an array is sufficient and easier to debug.
    return np.array([compute_basis_and_project(frame) for frame in sequence])