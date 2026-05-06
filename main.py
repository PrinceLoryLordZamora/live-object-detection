import os
import subprocess
import sys

# Pilitin ang installation ng libraries kapag hindi sila makita
try:
    import streamlit_webrtc
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-webrtc", "ultralytics", "opencv-python-headless", "av"])
    import streamlit_webrtc
    from ultralytics import YOLO

import streamlit as st

st.title("Object Detection is Ready!")
st.success("Na-install na ang lahat ng kailangan. Pwede mo nang ibalik ang original code mo.")
