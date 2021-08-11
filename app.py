
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import mediapipe as mp

st.markdown("This Application is Developed By Vijay")

DemoImage = "dwayne.jpg"

face_cascade = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

model_face_mesh = mp_face_mesh.FaceMesh()

drawing_spec = mp_drawing.DrawingSpec((0, 0, 225), thickness=1, circle_radius=1)

st.title("Face Recognition Application")
st.subheader("This application can perform various OpenCV Operations")
st.write()

add_dropbox = st.sidebar.selectbox(
    "Choose Input Source",
    ("Select", "Image processing", "Face Mesh", "Face Detect")
)

image = None
image_file_path = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file_path is not None:
    image = np.array(Image.open(image_file_path))
    # st.sidebar.image(image)
else:
    image = np.array(Image.open(DemoImage))
    st.write("Demo Image")
    # st.sidebar.image(image)
    st.image(image)

if add_dropbox == "Select":
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        st.image(image)
    st.write("Choose Input Source:")

# elif add_dropbox == "image processing":
#     add_lilbox2 = st.sidebar.selectbox(
#         "Choose any operation",
#         ("Select", "Grayscale", "Blue", "Green", "Red")
#     )
#     if add_lilbox2 == "Grayscale":
#         image_file_path = st.sidebar.file_uploader("Upload your file")
#         if image_file_path is not None:
#             image = np.array(Image.open(image_file_path))
#             st.sidebar.image(image)
#             gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             st.write("Your gray scaled image")
#             st.image(gray_image)
#
#     elif add_lilbox2 == "Green":
#         image_file_path = st.sidebar.file_uploader("Upload your file")
#         if image_file_path is not None:
#             image = np.array(Image.open(image_file_path))
#             st.sidebar.image(image)
#             zeros = np.zeros(image.shape[:2], dtype="uint8")
#             r, g, b = cv2.split(image)
#             green_image = cv2.merge([zeros, g, zeros])
#             st.image(green_image)
#
#     elif add_lilbox2 == "Blue":
#         image_file_path = st.sidebar.file_uploader("Upload your file")
#         if image_file_path is not None:
#             image = np.array(Image.open(image_file_path))
#             st.sidebar.image(image)
#             zeros = np.zeros(image.shape[:2], dtype="uint8")
#             r, g, b = cv2.split(image)
#             blue_image = cv2.merge([zeros, zeros, b])
#             st.image(blue_image)
#
#     elif add_lilbox2 == "Red":
#         image_file_path = st.sidebar.file_uploader("Upload your file")
#         if image_file_path is not None:
#             image = np.array(Image.open(image_file_path))
#             st.sidebar.image(image)
#             zeros = np.zeros(image.shape[:2], dtype="uint8")
#             r, g, b = cv2.split(image)
#             red_image = cv2.merge([r, zeros, zeros])
#             st.image(red_image)

elif add_dropbox == "Image processing":

    Filters = st.sidebar.radio("Choose among given operations:",
                               ("Grayscale", "Cartoon", "Canny Edge", "Contrast", "Brightness",
                                "Blur", "Blue", "Green", "Red")
                               )

    if Filters == "Grayscale":
        st.write("Converting to gray scale")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.write("Your gray scaled image")
        st.image(gray_image)

    elif Filters == "Cartoon":
        image = np.array(image)
        image = cv2.cvtColor(image, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Edges
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        # Color
        color = cv2.bilateralFilter(image, 9, 300, 300)
        # Cartoon
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        st.image(cartoon)

    elif Filters == "Canny Edge":
        st.write("Converting to canny")
        image = np.array(image)
        image = cv2.cvtColor(image, 1)
        image = cv2.GaussianBlur(image, (11, 11), 0)
        canny_image = cv2.Canny(image, 100, 150)
        st.image(canny_image)

    elif Filters == "Contrast":
        image = Image.fromarray(image)
        contrast_rate = st.slider("Choose Contrast", 0.5, 3.5)
        st.write("Converting image")
        enhancing = ImageEnhance.Contrast(image)
        contrasted_image = enhancing.enhance(contrast_rate)
        st.image(contrasted_image)

    elif Filters == "Brightness":
        image = Image.fromarray(image)
        brightness_rate = st.slider("Choose Brightness", 0.5, 3.5)
        st.write("Converting image")
        enhancing = ImageEnhance.Brightness(image)
        brightness_image = enhancing.enhance(brightness_rate)
        st.image(brightness_image)

    elif Filters == "Blur":
        image = Image.fromarray(image)
        image = np.array(image)
        blur_rate = st.slider("Choose Blur", 0.5, 3.5)
        st.write("Converting image")
        image = cv2.cvtColor(image, 1)
        blur_image = cv2.GaussianBlur(image, (11, 11), blur_rate)
        st.image(blur_image)

    elif Filters == "Blue":
        st.write("Converting to blue")
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        blue_image = cv2.merge([zeros, zeros, b])
        st.write("Your green image")
        st.image(blue_image)

    elif Filters == "Green":
        st.write("Converting to green")
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        green_image = cv2.merge([zeros, g, zeros])
        st.write("Your blue image")
        st.image(green_image)

    elif Filters == "Red":
        st.write("Converting to Red")
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        red_image = cv2.merge([r, zeros, zeros])
        st.write("Your red image")
        st.image(red_image)


elif add_dropbox == "Face Mesh":
    # image_file_path = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # if image_file_path is not None:
    #     image = np.array(Image.open(image_file_path))
    st.sidebar.image(image)
    results = model_face_mesh.process(image)

    for face_landmarks in results.multi_face_landmarks:
        # mp_drawing.draw_landmarks(image, face_landmarks, connections=mp_face_mesh.FACE_CONNECTIONS,
        #                           landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS, drawing_spec)
    st.write("Face Mesh")
    st.image(image)

elif add_dropbox == "Face Detect":
    # image_file_path = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # if image_file_path is not None:
    #     image = np.array(Image.open(image_file_path))
    st.sidebar.image(image)
    new_img = np.array(image)
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 2)
    st.write("Detecting faces")
    st.image(img)
