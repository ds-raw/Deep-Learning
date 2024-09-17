# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:36:22 2024

@author: daffa
"""

from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2

model = YOLO(r"C:\Users\daffa\Downloads\best car detect clb.pt")
names = model.model.names

cap = cv2.VideoCapture(r"C:\Users\daffa\Videos\Captures\0917.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(r"C:\Users\daffa\Videos\Captures\091723.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

line_pts = [(300, 700), (1280, 700)]

# Initialize SpeedEstimator with all necessary arguments directly
speed_obj = speed_estimation.SpeedEstimator(names=names, reg_pts=line_pts, view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
