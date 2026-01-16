import numpy as np

class HandGraph:
    """
    Defines the physical structure of the hand for GCN initialization.
    Strategy: Spatial Configuration Partitioning (Standard in ST-GCN).
    """
    def __init__(self, strategy='spatial'):
        self.num_node = 21
        self.self_link = [(i, i) for i in range(self.num_node)]
        
        # MediaPipe Connectivity (0-indexed)
        # Wrist(0) -> Thumb(1,2,3,4), Index(5,6,7,8), etc.
        neighbor_link = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        self.edge = self.self_link + neighbor_link
        self.center = 0 # Wrist is the root

        if strategy == 'spatial':
            self.A = self.get_spatial_graph(self.num_node, self.self_link, neighbor_link)
        else:
            raise ValueError("Only 'spatial' partitioning is supported for this architecture.")

    def get_spatial_graph(self, num_node, self_link, neighbor_link):
        I = self_link
        # Outward connections (Root -> Tips)
        outward = neighbor_link 
        # Inward connections (Tips -> Root)
        inward = [(j, i) for (i, j) in neighbor_link]
        
        # Matrix construction
        A = np.zeros((3, num_node, num_node))
        
        # Subset 0: The Node itself (Stationary)
        for i, j in I:
            A[0, i, j] = 1 
            
        # Subset 1: Centripetal (Tips moving to Root)
        for i, j in inward:
            A[1, i, j] = 1
            
        # Subset 2: Centrifugal (Root pushing to Tips)
        for i, j in outward:
            A[2, i, j] = 1
            
        # Normalize (Row-stochastic) to prevent feature explosion
        for i in range(3):
            row_sum = np.sum(A[i], axis=1)
            # Avoid division by zero
            row_sum[row_sum == 0] = 1 
            A[i] = A[i] / row_sum[:, np.newaxis]
            
        return A