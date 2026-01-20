import numpy as np

class HandGraph:
    """
    Dual-Hand Graph Structure.
    Nodes 0-20: Left Hand
    Nodes 21-41: Right Hand
    """
    def __init__(self, strategy='spatial'):
        self.num_node = 42
        self.self_link = [(i, i) for i in range(self.num_node)]
        
        # Define Single Hand Connections (MediaPipe Standard)
        base_hand = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        # Left Hand (0-20)
        left_hand_edges = base_hand
        # Right Hand (21-41) -> Add 21 to all indices
        right_hand_edges = [(i + 21, j + 21) for (i, j) in base_hand]
        
        neighbor_link = left_hand_edges + right_hand_edges
        
        self.edge = self.self_link + neighbor_link
        self.center = 0 # Left Wrist as primary root

        if strategy == 'spatial':
            self.A = self.get_spatial_graph(self.num_node, self.self_link, neighbor_link)

    def get_spatial_graph(self, num_node, self_link, neighbor_link):
        I = self_link
        outward = neighbor_link 
        inward = [(j, i) for (i, j) in neighbor_link]
        A = np.zeros((3, num_node, num_node))
        for i, j in I: A[0, i, j] = 1 
        for i, j in inward: A[1, i, j] = 1
        for i, j in outward: A[2, i, j] = 1
        for i in range(3):
            row_sum = np.sum(A[i], axis=1)
            row_sum[row_sum == 0] = 1 
            A[i] = A[i] / row_sum[:, np.newaxis]
        return A