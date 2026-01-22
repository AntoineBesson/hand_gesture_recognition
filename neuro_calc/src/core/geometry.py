import numpy as np

# --- AJOUTEZ CETTE FONCTION ---
def process_dual_hand_frame(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """
    Prend deux mains (21, 3), canonicalise la SCÈNE unifiée et renvoie (42, 3).
    Gère les mains manquantes (remplissage par des zéros).
    """
    # 1. Gestion des mains manquantes ou vides
    if left_hand is None or left_hand.size == 0 or np.all(left_hand == 0):
        left_hand = np.zeros((21, 3))
    if right_hand is None or right_hand.size == 0 or np.all(right_hand == 0):
        right_hand = np.zeros((21, 3))

    has_left = np.any(left_hand)
    has_right = np.any(right_hand)

    # 2. Logique de projection selon le cas
    if has_left and has_right:
        # --- MODE DOUBLE MAIN ---
        # Centre: Point médian entre les poignets
        wrist_l = left_hand[0]
        wrist_r = right_hand[0]
        center = (wrist_l + wrist_r) / 2.0
        
        # Translation vers l'origine
        l_centered = left_hand - center
        r_centered = right_hand - center
        
        # Rotation: Aligner le vecteur Poignet-Poignet avec l'axe X
        v_lr = wrist_r - wrist_l
        norm_v_lr = np.linalg.norm(v_lr)
        x_axis = v_lr / (norm_v_lr + 1e-6)
        
        # Axe Y approximatif (moyenne des vecteurs "Haut" des mains)
        v_up_l = left_hand[9] - left_hand[0]
        v_up_r = right_hand[9] - right_hand[0]
        y_approx = (v_up_l + v_up_r) / 2.0
        
        # Orthogonalisation (Gram-Schmidt)
        z_axis = np.cross(x_axis, y_approx)
        z_axis /= (np.linalg.norm(z_axis) + 1e-6)
        
        y_axis = np.cross(z_axis, x_axis) # Déjà normalisé
        
        # Matrice de rotation R
        R = np.stack([x_axis, y_axis, z_axis])
        
        # Projection
        l_proj = np.dot(l_centered, R.T)
        r_proj = np.dot(r_centered, R.T)
        
        # Normalisation d'échelle (basée sur la taille moyenne des mains)
        scale = (np.linalg.norm(v_up_l) + np.linalg.norm(v_up_r)) / 2.0
        
        return np.concatenate([l_proj, r_proj], axis=0) / (scale + 1e-6)

    elif has_right:
        # --- MODE MAIN DROITE SEULE ---
        # On utilise l'ancienne logique pour la droite, zéros pour la gauche
        return np.concatenate([np.zeros((21,3)), _canonicalize_single(right_hand)], axis=0)
        
    elif has_left:
        # --- MODE MAIN GAUCHE SEULE ---
        return np.concatenate([_canonicalize_single(left_hand), np.zeros((21,3))], axis=0)
        
    else:
        # --- VIDE ---
        return np.zeros((42, 3))

def _canonicalize_single(hand):
    """Fonction helper pour une seule main (Logique originale)"""
    wrist = hand[0]
    centered = hand - wrist
    
    v_primary = centered[9]
    norm_primary = np.linalg.norm(v_primary)
    y_axis = v_primary / (norm_primary + 1e-6)
    
    v_secondary = centered[17]
    z_axis = np.cross(y_axis, v_secondary)
    z_axis /= (np.linalg.norm(z_axis) + 1e-6)
    
    x_axis = np.cross(y_axis, z_axis)
    
    R = np.stack([x_axis, y_axis, z_axis])
    
    # Projection et normalisation par la taille de la main
    return np.dot(centered, R.T) / (norm_primary + 1e-6)

# Gardez votre ancienne fonction compute_basis_and_project si besoin, 
# mais process_dual_hand_frame est celle utilisée par le dataset.