import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp


#selfie segmentation - background separation with color and image

#Drawing utility, used to draw the face mesh on to the video frame
mp_drawing = mp.solutions.drawing_utils
#Face mesh utility
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#drawing specifications
drawing_spec = mp_drawing.DrawingSpec((0, 225, 225), thickness=1, circle_radius=1)

#Model for face mesh
model_ss = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

#background image
#bg_image = None
bg_image = cv2.imread("vegita.jpg")

capture = cv2.VideoCapture(0)

while capture.isOpened():
    flag, frame = capture.read()
    if not flag:
        print('Could not access the webcam')

    results = model_ss.process(frame)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
    if bg_image is None:
        bg_image = np.zeros(frame.shape, dtype = np.uint8)
        bg_image[:] = (0, 255, 0)
    bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
    #cv2.imshow('Frame', results.segmentation_mask)    output_image = np.where(condition, frame, bg_image)

    cv2.imshow('Frame', output_image)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()