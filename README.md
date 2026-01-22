# hand_gesture_recognition

## Motivations
This project is my second time tackling an image/video ML model. In my previous project (license plate recognition), I used CNN twice to find the characters on the license plate, then used ML to recognize the characters. I had used a public dataset. 

For this project, I used mediapipe for hand recognition. The vertices and nodes are then normalised and projected in the same plane. I created my own dataset, 21 file per character. The model works pretty accurately, despite being pretty slow (the confidence interval could be widened to fasten the recognition, but we might lose accuracy). 

## How to use it yourself

1. Clone the repo and install dependencies:
```
cd neuro_calc
poetry install
```

2. Record your own hand gesture data (if you don't want to, skip to step 4 to use my model):
```
poetry run python tools/recorder.py
```
Use SPACE to record, N/P to switch classes, Q to quit.

3. Train the model:
```
poetry run python train.py
```

4. Run live inference:
```
poetry run python inference.py
```
