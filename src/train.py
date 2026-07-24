
import os
import yaml
import urllib.request
from ultralytics import YOLO

def train():
    dataset_dir = "/content/dataset"
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    weights_path = "yolov8n.pt"

    # Pre-download base weights if missing
    if not os.path.exists(weights_path):
        print("⬇️ Downloading base model weights (yolov8n.pt)...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
        urllib.request.urlretrieve(url, weights_path)

    # Initialize and train
    model = YOLO(weights_path)
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project="leaf_disease_run",
        name="train",
        exist_ok=True
    )
    print("✅ Training complete!")

if __name__ == "__main__":
    train()
