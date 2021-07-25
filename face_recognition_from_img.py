import mediapipe
import numpy
import cv2
import face_recognition

image_train = face_recognition.load_image_file('images/dwayne1.jpg')
image_encodings_train = face_recognition.face_encodings(image_train)[0]
image_locations_train = face_recognition.face_locations(image_train)[0]

image_test = face_recognition.load_image_file('images/dwayne2.jpg')
image_encodings_test = face_recognition.face_encodings(image_test)[0]

results = face_recognition.compare_faces([image_encodings_train], image_encodings_test)[0]
dst = face_recognition.face_distance([image_encodings_train], image_encodings_test)
#print(results, image_locations_train)        #this will give us the top,right,down,left locations

if results:
    image_train = cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
    cv2.rectangle(image_train,
                    (image_locations_train[3], image_locations_train[0]),
                    (image_locations_train[1], image_locations_train[2]),
                    (0, 225, 0),
                    2)
    cv2.putText(image_train, f'{results} {dst}',
                (60, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 225, 0),
                1)
    cv2.imshow("dwayne", image_train)
else:
    print("Coundn't recognize the face. Result was {results} and distance was {dst}")

cv2.waitKey(0)

#image = cv2.imread('images/vegita.jpg')
#cv2.imshow("vegita", image)
#cv2.waitKey(0)