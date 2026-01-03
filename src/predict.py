from ultralytics import YOLO
import os

from utils import load_config
    
def predict():
    # Load configuration
    cfg = load_config()
    model_path = cfg['weights']
    
    # Load the model
    # Note: Replace 'yolo11s.pt' with your trained weight path, e.g., 'runs/train/exp/weights/best.pt'
    model = YOLO(model_path) 

    # Predict
    source = "datasets/Safety-Helmet-Wearing-Dataset.v3-base-dataset.yolov11/test/images" # Source directory
    if not os.path.exists(source):
        print(f"Source directory {source} not found. Testing with a dummy image if possible or skip.")
        return

    results = model.predict(source=source, save=True, conf=0.25)
    print(f"Prediction complete. Results saved to {results[0].save_dir}")

if __name__ == '__main__':
    predict()
