import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
from utils import load_config
import os
import sys

class DesktopDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("工地安全帽检测系统 (Safety Helmet Detection)")
        self.root.geometry("1400x800")
        
        # Load Config
        try:
            self.cfg = load_config()
            self.default_model = self.cfg.get('weights', 'yolo11s.pt')
        except:
            self.default_model = "yolo11s.pt"

        self.model = None
        self.thread = None
        self.stop_event = threading.Event()
        self.current_image = None # Helper for re-inference
        
        self.setup_ui()
        
        # Load default model after UI setup
        self.load_model(self.default_model)

    def setup_ui(self):
        # LEFT SIDEBAR
        sidebar = tk.Frame(self.root, width=300, bg='#f0f0f0', padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Title
        tk.Label(sidebar, text="配置面板", font=("Arial", 16, "bold"), bg='#f0f0f0').pack(pady=(0, 20))
        
        # Model
        labelframe_model = tk.LabelFrame(sidebar, text="模型设置 (Model)", bg='#f0f0f0', padx=5, pady=5)
        labelframe_model.pack(fill=tk.X, pady=5)
        
        self.lbl_model_path = tk.Label(labelframe_model, text=self.default_model, wraplength=250, bg='#f0f0f0', fg="blue")
        self.lbl_model_path.pack(pady=5)
        tk.Button(labelframe_model, text="选择模型文件 (Load .pt)", command=self.select_model).pack(fill=tk.X)
        
        # Confidence
        labelframe_conf = tk.LabelFrame(sidebar, text="置信度 (Confidence)", bg='#f0f0f0', padx=5, pady=5)
        labelframe_conf.pack(fill=tk.X, pady=5)
        
        self.var_conf = tk.DoubleVar(value=0.25)
        self.lbl_conf = tk.Label(labelframe_conf, text="0.25", bg='#f0f0f0')
        self.lbl_conf.pack()
        scale = tk.Scale(labelframe_conf, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, 
                         variable=self.var_conf, command=self.on_conf_change)
        scale.pack(fill=tk.X)

        # Input Mode
        labelframe_input = tk.LabelFrame(sidebar, text="输入模式 (Input)", bg='#f0f0f0', padx=5, pady=5)
        labelframe_input.pack(fill=tk.X, pady=5)
        
        self.combo_mode = ttk.Combobox(labelframe_input, values=["图片检测 (Image)", "视频检测 (Video)", "摄像头实时 (Webcam)"], state="readonly")
        self.combo_mode.current(0)
        self.combo_mode.pack(fill=tk.X, pady=5)
        self.combo_mode.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        self.btn_open = tk.Button(labelframe_input, text="打开文件 (Open File)", command=self.open_file)
        self.btn_open.pack(fill=tk.X, pady=2)
        
        self.btn_cam = tk.Button(labelframe_input, text="开启摄像头 (Start Cam)", command=self.toggle_cam)
        self.btn_cam.pack(fill=tk.X, pady=2)
        self.btn_cam.pack_forget() # Hide initially

        # Status
        self.lbl_status = tk.Label(sidebar, text="Ready", bg='#f0f0f0', fg="gray")
        self.lbl_status.pack(side=tk.BOTTOM)

        # RIGHT DISPLAY
        self.display_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.lbl_image = tk.Label(self.display_frame, text="请选择输入 (Please select input)", bg="#2b2b2b", fg="white")
        self.lbl_image.pack(expand=True)

    def load_model(self, path):
        try:
            self.lbl_status.config(text=f"Loading model: {os.path.basename(path)}...")
            self.root.update()
            self.model = YOLO(path)
            self.lbl_model_path.config(text=os.path.basename(path))
            self.lbl_status.config(text="Model loaded successfully.")
            messagebox.showinfo("Success", "模型加载成功!")
        except Exception as e:
            self.lbl_status.config(text="Model load failed.")
            messagebox.showerror("Error", f"模型加载失败:\n{str(e)}")

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt")])
        if path:
            self.load_model(path)

    def on_conf_change(self, val):
        self.lbl_conf.config(text=f"{float(val):.2f}")
        # Build-in debounce or just direct update for image
        if self.combo_mode.get() == "图片检测 (Image)" and self.current_image is not None:
             self.inference_image(self.current_image)

    def on_mode_change(self, event):
        self.stop_video()
        mode = self.combo_mode.get()
        if "Webcam" in mode:
            self.btn_open.pack_forget()
            self.btn_cam.pack(fill=tk.X, pady=2)
            self.lbl_image.config(text="点击开启摄像头", image="")
        else:
            self.btn_cam.pack_forget()
            self.btn_open.pack(fill=tk.X, pady=2)
            self.lbl_image.config(text="请选择文件", image="")

    def open_file(self):
        mode = self.combo_mode.get()
        if "Image" in mode:
            path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
            if path:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.current_image = img
                self.inference_image(img)
        elif "Video" in mode:
            path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mkv")])
            if path:
                self.start_video(path)

    def toggle_cam(self):
        if self.thread and self.thread.is_alive():
            self.stop_video()
            self.btn_cam.config(text="开启摄像头 (Start Cam)")
        else:
            self.start_video(0)
            self.btn_cam.config(text="停止 (Stop)")

    def inference_image(self, img_bgr):
        if self.model is None: return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, conf=self.var_conf.get())
        res_plotted = results[0].plot()
        self.show_image(res_plotted)

    def start_video(self, source):
        self.stop_video()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.video_loop, args=(source,), daemon=True)
        self.thread.start()

    def stop_video(self):
        if self.thread:
            self.stop_event.set()
            self.thread.join(timeout=1.0)
            self.thread = None

    def video_loop(self, source):
        cap = cv2.VideoCapture(source)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
                    continue
                else: 
                    break
            
            if self.model:
                results = self.model(frame, conf=self.var_conf.get())
                res_plotted = results[0].plot()
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            # Update UI in main thread safely? 
            # Tkinter is not thread safe but usually Image update works if handled carefully or via after()
            # For simplicity/speed we will try invoking valid update
            self.root.after(0, self.show_image, img_rgb)
            
            time.sleep(0.01)
        cap.release()

    def show_image(self, img_array):
        # img_array is RGB numpy array
        h, w, _ = img_array.shape
        # Resize to fit display
        display_w = self.display_frame.winfo_width()
        display_h = self.display_frame.winfo_height()
        
        if display_w < 10 or display_h < 10: return
        
        scale = min(display_w/w, display_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        img = Image.fromarray(img_array)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=photo, text="")
        self.lbl_image.image = photo # Keep reference

def main():
    root = tk.Tk()
    app = DesktopDemoApp(root)
    
    # Handle window close
    def on_closing():
        app.stop_video()
        root.destroy()
        sys.exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
