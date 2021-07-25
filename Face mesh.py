import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp


#Face mesh

#Drawing utility, used to draw the face coordinate on to the video frame
mp_drawing = mp.solutions.drawing_utils
#Face mesh
mp_face_mesh = mp.solutions.face_mesh

#drawing specifications
drawing_spec = mp_drawing.DrawingSpec((225, 225, 0), thickness=1, circle_radius=1)

#Model for face mesh
model_facemesh = mp_face_mesh.FaceMesh()

capture = cv2.VideoCapture(0)

while capture.isOpened():
    flag, frame = capture.read()
    if not flag:
        print('Could not access the webcam')
        break

    results = model_facemesh.process(frame)
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks:
            #print(landmark)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()