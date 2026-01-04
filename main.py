
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import os
import random

size_df = pd.read_csv( r"C:\Users\sidda\OneDrive\Desktop\HACKATHON\Body Measurements _ original_CSV.csv")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)
PIXEL_TO_CM = 0.15

def distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def recommend_size(shoulder_cm):
    for _, row in size_df.iterrows():
        if row['shoulder_min'] <= shoulder_cm <= row['shoulder_max']:
            return row['size']
    return "M"

def load_random_apparel(size):
    folder = os.path.join("apparel", size)
    imgs = [f for f in os.listdir(folder) if f.endswith(".png")]
    if not imgs:
        return None
    path = os.path.join(folder, random.choice(imgs))
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        h, w, _ = frame.shape

        shoulder_px = distance(
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            w, h
        )
        shoulder_cm = round(shoulder_px * PIXEL_TO_CM, 1)
        size = recommend_size(shoulder_cm)

        apparel = load_random_apparel(size)
        if apparel is not None:
            aw = int(shoulder_px * 1.4)
            ah = int(aw * 1.6)
            apparel = cv2.resize(apparel, (aw, ah))

            x = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
            y = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)

            for i in range(ah):
                for j in range(aw):
                    if apparel[i, j][3] > 0 and y+i < h and x+j < w:
                        frame[y+i, x+j] = apparel[i, j][:3]

        cv2.putText(frame, f"Shoulder: {shoulder_cm} cm", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Size: {size}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Smart Fitting Room", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
