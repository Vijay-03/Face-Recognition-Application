
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

face_cascade = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.name = "Cheers"

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        name = self.name
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (95, 207, 30), 3)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (95, 207, 30), -1)
            cv2.putText(frame, str(name), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img


DemoImage = "dwayne.jpg"

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

model_face_mesh = mp_face_mesh.FaceMesh()

drawing_spec = mp_drawing.DrawingSpec((0, 0, 225), thickness=1, circle_radius=1)

heading = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
          ' font-size: 20px;">Developed by Vijay</p'
st.write(heading, unsafe_allow_html=True)
st.subheader("This application can perform various Computer Vision Operations")
st.write()

add_dropbox = st.sidebar.selectbox(
    "Choose Input Source",
    ("Select", "Live Face Detection", "Image processing", "Face Mesh", "Face Detect")
)

image = None
image_file_path = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file_path is not None:
    image = np.array(Image.open(image_file_path))
else:
    image = np.array(Image.open(DemoImage))

if add_dropbox == "Select":
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        st.image(image)
    else:
        dm = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
               ' font-size: 20px;">Demo Image:</p'
        st.write(dm, unsafe_allow_html=True)
        # st.sidebar.image(image)
        st.image(image)

    text = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
           ' font-size: 20px;">Choose Input Source:</p'
    st.write(text, unsafe_allow_html=True)
    # st.write("Choose Input Source:")

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
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting to gray scale</p'
        st.write(cv, unsafe_allow_html=True)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         st.write("Your gray scaled image")  
        st.image(gray_image)

    elif Filters == "Cartoon":
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting to Cartoon</p'
        st.write(cv, unsafe_allow_html=True)
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
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting to Canny Edge</p'
        st.write(cv, unsafe_allow_html=True)
        image = np.array(image)
        image = cv2.cvtColor(image, 1)
        image = cv2.GaussianBlur(image, (11, 11), 0)
        canny_image = cv2.Canny(image, 100, 150)
        st.image(canny_image)

    elif Filters == "Contrast":
        image = Image.fromarray(image)
        contrast_rate = st.slider("Choose Contrast", 0.5, 3.5)
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting image</p'
        st.write(cv, unsafe_allow_html=True)
        enhancing = ImageEnhance.Contrast(image)
        contrasted_image = enhancing.enhance(contrast_rate)
        st.image(contrasted_image)

    elif Filters == "Brightness":
        image = Image.fromarray(image)
        brightness_rate = st.slider("Choose Brightness", 0.5, 3.5)
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting image</p'
        st.write(cv, unsafe_allow_html=True)
        enhancing = ImageEnhance.Brightness(image)
        brightness_image = enhancing.enhance(brightness_rate)
        st.image(brightness_image)

    elif Filters == "Blur":
        image = Image.fromarray(image)
        image = np.array(image)
        blur_rate = st.slider("Choose Blur", 0.5, 3.5)
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting image</p'
        st.write(cv, unsafe_allow_html=True)
        image = cv2.cvtColor(image, 1)
        blur_image = cv2.GaussianBlur(image, (11, 11), blur_rate)
        st.image(blur_image)

    elif Filters == "Blue":
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting to Blue</p'
        st.write(cv, unsafe_allow_html=True)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        blue_image = cv2.merge([zeros, zeros, b])
        st.write("Your blue image")
        st.image(blue_image)

    elif Filters == "Green":
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting to Green</p'
        st.write(cv, unsafe_allow_html=True)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        green_image = cv2.merge([zeros, g, zeros])
        st.write("Your green image")
        st.image(green_image)

    elif Filters == "Red":
        cv = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
             ' font-size: 20px;">Converting to Red</p'
        st.write(cv, unsafe_allow_html=True)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        red_image = cv2.merge([r, zeros, zeros])
        st.write("Your red image")
        st.image(red_image)


elif add_dropbox == "Face Mesh":
    st.sidebar.image(image)
    results = model_face_mesh.process(image)

    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS, drawing_spec)
    fm = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
         ' font-size: 20px;">Face Mesh</p'
    st.write(fm, unsafe_allow_html=True)
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
    fd = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
         ' font-size: 20px;">Detecting faces</p'
    st.write(fd, unsafe_allow_html=True)
    st.image(img)
    
elif add_dropbox == "Live Face Detection":
    message = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
              ' font-size: 20px;">Live Face Detection</p'
    st.write(message, unsafe_allow_html=True)
    # st.subheader("Live Face Detection")
    drawing_spec = mp_drawing.DrawingSpec((225, 225, 0), thickness=1, circle_radius=1)

    st.sidebar.markdown("---")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
