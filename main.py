import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from ultralytics import YOLO
import av

# 1. I-load ang YOLO model sa simula pa lang
# Gagamit tayo ng 'yolov8n.pt' (nano) para mabilis at hindi mabigat sa server
model = YOLO("yolov8n.pt") 

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # I-convert ang frame sa image format na gamit ng OpenCV
        img = frame.to_ndarray(format="bgr24")

        # 2. Gawin ang Object Detection
        results = model.predict(img, conf=0.5) # 0.5 confidence threshold

        # 3. I-draw ang results (boxes at labels) sa image
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def main():
    st.set_page_config(page_title="YOLOv8 WebRTC", layout="centered")
    
    st.title("🛡️ YOLOv8 Real-time Detector")
    st.write("Live detection gamit ang iyong camera.")

    # 4. WebRTC Streamer setup
    webrtc_streamer(
        key="yolo-detection",
        video_frame_callback=lambda frame: YOLOVideoTransformer().transform(frame),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

    st.sidebar.header("Instructions")
    st.sidebar.info(
        "1. I-allow ang camera access.\n"
        "2. Hintayin mag-load ang model.\n"
        "3. Magpakita ng objects (tao, cellphone, laptop, etc.) sa camera."
    )

if __name__ == "__main__":
    main()
