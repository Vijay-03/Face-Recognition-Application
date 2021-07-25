import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp


#Face detection

#Drawing utility, used to draw the face coordinate on to the video frame
mp_drawing = mp.solutions.drawing_utils
#Face detection utility
mp_face_detectiion = mp.solutions.face_detection

#Model for detecting the face coordinates
model_detection = mp_face_detectiion.FaceDetection()

capture = cv2.VideoCapture(0)

while capture.isOpened():
    flag, frame = capture.read()
    if not flag:
        print('Could not access the webcam')
        break

    results = model_detection.process(frame)
    if results.detections:
        for landmark in results.detections:
            mp_drawing.draw_detection(frame, landmark)
        #print(results.detections)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()