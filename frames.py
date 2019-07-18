import cv2
import os

DIR = "/home/panchenko/Downloads/obstacle_detection/video/"

id = 0
for filename in os.listdir(DIR):
    if not filename.endswith(".avi"):
        continue
    cap = cv2.VideoCapture(DIR + filename)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(DIR, 'frames', str(id) + ".jpg"), frame)
        id += 1