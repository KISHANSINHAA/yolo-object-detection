import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Object Detection App", layout="wide")

st.title("Real-Time Computer Vision Object Detection System")
st.write("Upload an image or use your webcam for real-time detection.")

# Load YOLO model
model = YOLO("model/yolov8n.pt")

# ---------------- IMAGE UPLOAD ---------------- #

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_array = np.array(image)

    results = model(img_array)
    result = results[0]

    output = result.plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Detection Result")
        st.image(output, width="stretch")

    boxes = result.boxes

    if boxes is not None:

        names = model.names
        classes = boxes.cls.tolist()
        confidences = boxes.conf.tolist()

        data = []

        for cls, conf in zip(classes, confidences):
            label = names[int(cls)]
            data.append({
                "Object": label,
                "Confidence": round(conf, 3)
            })

        df = pd.DataFrame(data)

        st.subheader("Detection Results")
        st.dataframe(df, use_container_width=True)

        st.subheader("Object Count")

        count_dict = df["Object"].value_counts()

        for obj, count in count_dict.items():
            st.write(f"{obj}: {count}")

# ---------------- WEBCAM DETECTION ---------------- #

st.divider()
st.subheader("Live Webcam Detection")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model(img)
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback
)
