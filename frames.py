import cv2
import os

DIR = "/home/serg/PycharmProjects/image_processing/video/camera_1/"

id = 0
for filename in os.listdir(DIR):
    if not filename.endswith(".mp4"):
        continue
    cap = cv2.VideoCapture(DIR + filename)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (720, 360))
        frame = frame[220:, :434]
        cv2.imwrite(os.path.join(DIR, 'frames', str(id) + ".jpg"), frame)
        id += 1