import cv2
import os

filename = "/home/serg/PycharmProjects/image_processing/video/20190717_132436.mp4"
DIR = '/home/serg/PycharmProjects/image_processing/video/camera_2'

id = 0

cap = cv2.VideoCapture(filename, )

if (cap.isOpened()== False):
  print("Error opening video stream or file")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (720, 360))
    frame = frame[220:, :434]
    cv2.imwrite(os.path.join(DIR, str(id) + ".jpg"), frame)
    id += 1