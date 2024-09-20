from pathlib import Path

import PIL.Image
from src.config import *
import streamlit as st
import helper
import PIL

## setting the page layout
st.set_page_config(
    page_title="Object detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

#heading of main page
st.title("Object detection and Tracking using YOLOv8")

#sidebar
st.sidebar.header('ML Model Config')

#Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40))/100

##selecting Detection and Segmentation
if model_type == 'Detection':
    model_path = Path(YOLO_MODEL_DIR)
elif model_type == 'Segmentation':
    model_path = Path(YOLO_MODEL_SEG_DIR)

#Load the Pre_train ML models
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check specific path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST)

source_img = None
#if image is selected
if source_radio == IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("Jpg", "jpeg", "png", "bmp", "webp"))
    
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(NON_DETECTED_IMAGE_DIR)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path,caption="Default Image" ,use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occure while opening the image.")
            st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(DETECTED_IMAGE_DIR)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path,caption='Detected Image', use_column_width=True)

            else:
                if st.sidebar.button('Detect Objects'):
                    res=model.predict(uploaded_image,conf=confidence)
                    boxes= res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption="Detected Image", use_column_width=True)

                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    
                    except Exception as ex:
                        st.write("No image uploaded yet!")

elif source_radio == VIDEO:
    helper.play_stored_video(confidence,model)

elif source_radio == WEBCAM:
    helper.play_webcam(confidence,model)

elif source_radio == RTSP:
    helper.play_rtsp_stream(confidence,model)

elif source_radio == YOUTUBE:
    helper.play_youtube_video(confidence,model)

else:
    st.error("Please select the valid source type!")