# hand_gesture_recognition
neuro_calc/
├── conf/                   # [Hydra] Structured configs (YAML)
│   ├── base/
│   │   └── camera.yaml     # Sensor intrinsics/extrinsics
│   ├── model/
│   │   └── st_gcn.yaml     # Architecture hyperparameters
│   └── config.yaml         # Main entry point
├── data/
│   ├── raw/                # Immutable raw video dumps
│   └── processed/          # Canonicalized .npy sequences (Graph Inputs)
├── notebooks/              # For EDA and prototype visualization only
├── src/
│   ├── __init__.py
│   ├── core/               # PURE MATHEMATICS (No heavy dependencies)
│   │   ├── geometry.py     # SO(3) projections, vector calc
│   │   └── signal.py       # Temporal smoothing (Savitzky-Golay filters)
│   ├── pipeline/           # DATA INGESTION
│   │   ├── sensor.py       # Webcam/MediaPipe wrappers
│   │   └── preprocessor.py # Orchestrates canonicalization
│   └── models/             # NEURAL NETWORKS
│       ├── components/     # Layers (GCN blocks, Attention heads)
│       └── solver.py       # The RPN calculator logic
├── tests/                  # Pytest suite (Crucial for geometric unit tests)
├── pyproject.toml          # Poetry/Setuptools configuration
└── main.py                 # CLI Entry point (minimal logic)