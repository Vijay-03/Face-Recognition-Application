import mediapipe
import numpy as np
import cv2
import face_recognition

dwayne = face_recognition.load_image_file('images/dwayne1.jpg')
dwayne_encodings = face_recognition.face_encodings(dwayne)[0]

kevin = face_recognition.load_image_file('images/kevin1.jpg')
kevin_encodings = face_recognition.face_encodings(kevin)[0]

known_face_encodings = [dwayne_encodings, kevin_encodings]
known_face_names = ['Dwayne', 'Kevin Hart']

capture = cv2.VideoCapture(0)

while capture.isOpened():
    flag, frame = capture.read()
    if not flag:
        print("Couldn't access camera.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=1/4, fy=1/4)
    #rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    #print(face_encodings)
    face_names = []
    for face_encoding in face_encodings:
        #print(len(face_encoding))
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'Unknown'
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        print(face_names)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 0, 225), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

    #cv2.imshow('Frame', small_frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()