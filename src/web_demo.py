import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
from utils import load_config

# Load config
try:
    cfg = load_config()
    default_model = cfg['weights']
except:
    default_model = "yolo11s.pt"

# Config
st.set_page_config(layout="wide", page_title="工地安全帽检测系统", page_icon="👷")

def main():
    st.title("👷 工地安全帽佩戴检测系统")
    st.sidebar.header("配置 (Settings)")

    # Model Selection
    model_path = st.sidebar.text_input("模型路径 (Model Path)", default_model)
    conf_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25)
    
    try:
        model = YOLO(model_path)
        st.sidebar.success("模型加载成功!")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return

    # Input Mode
    mode = st.sidebar.selectbox("检测模式 (Mode)", ["图片检测 (Image)", "视频检测 (Video)", "摄像头实时 (Webcam)"])

    if mode == "图片检测 (Image)":
        uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="原始图片", use_container_width=True)
            
            if st.button("开始检测"):
                # Inference
                res = model.predict(image, conf=conf_threshold)
                res_plotted = res[0].plot()
                st.image(res_plotted, caption="检测结果", channels="BGR", use_container_width=True)

    elif mode == "视频检测 (Video)":
        uploaded_file = st.file_uploader("上传视频", type=['mp4', 'avi'])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference
                res = model.predict(frame, conf=conf_threshold)
                res_plotted = res[0].plot()
                
                stframe.image(res_plotted, channels="BGR", use_container_width=True)
            cap.release()

    elif mode == "摄像头实时 (Webcam)":
        st.warning("Webcam mode works best locally. Ensure camera access.")
        run = st.checkbox('开启摄像头')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            if frame is None:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, conf=conf_threshold)
            annotated_frame = results[0].plot()
            
            FRAME_WINDOW.image(annotated_frame, channels="BGR")
        else:
            camera.release()

if __name__ == "__main__":
    main()
