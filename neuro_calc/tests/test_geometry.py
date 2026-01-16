import sys
import os

# dynamic path: Add the parent directory (neuro_calc) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.core.geometry import compute_basis_and_project

def generate_synthetic_hand():
    """
    Creates a perfectly flat 'hand' in the XY plane.
    Wrist at (0,0,0). Middle finger along Y axis.
    """
    # Initialize 21 points at zero
    hand = np.zeros((21, 3))
    
    # Set Middle MCP (Index 9) to be at (0, 1, 0)
    # This defines our "canonical" Up vector
    hand[9] = [0, 1, 0]
    
    # Set Pinky MCP (Index 17) to be at (1, 0, 0)
    # This defines our "canonical" Side vector
    hand[17] = [1, 0, 0]
    
    # Add random noise to other joints to simulate a real shape
    # (We don't care about biological plausibility, only geometric rigidity)
    rng = np.random.default_rng(42)
    hand[1:] += rng.normal(0, 0.1, size=(20, 3))
    
    return hand

def random_rotation_matrix():
    """
    Generates a random 3x3 rotation matrix (SO(3)).
    """
    # Generate a random unit vector (axis of rotation)
    rng = np.random.default_rng()
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    
    # Random angle
    theta = rng.uniform(0, 2 * np.pi)
    
    # Rodrigues' formula for rotation matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def test_rotation_invariance():
    """
    The Core Proof: Canonical(Hand) == Canonical(R * Hand + T)
    """
    # 1. Source Data
    base_hand = generate_synthetic_hand()
    
    # 2. Perturbation (Simulate User moving hand)
    R = random_rotation_matrix()
    T = np.random.uniform(-10, 10, size=(1, 3)) # Random Translation
    
    # Apply affine transformation: P' = P @ R.T + T
    transformed_hand = (base_hand @ R.T) + T
    
    # 3. Execution
    canonical_base = compute_basis_and_project(base_hand)
    canonical_trans = compute_basis_and_project(transformed_hand)
    
    # 4. Assertion
    # We expect the outputs to be identical (within floating point tolerance)
    # This proves the model "sees" the same hand regardless of rotation.
    np.testing.assert_allclose(
        canonical_base, 
        canonical_trans, 
        atol=1e-6, 
        err_msg="The canonicalization is not rotation invariant!"
    )

def test_translation_invariance():
    """
    Simple check: Moving the hand in space should not change output.
    """
    base_hand = generate_synthetic_hand()
    T = np.array([[100, -50, 25]])
    
    translated_hand = base_hand + T
    
    canonical_base = compute_basis_and_project(base_hand)
    canonical_trans = compute_basis_and_project(translated_hand)
    
    np.testing.assert_allclose(canonical_base, canonical_trans, atol=1e-6)