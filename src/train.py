import os
import yaml
import urllib.request
from ultralytics import YOLO

def train():
    dataset_dir = "/content/dataset"
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    weights_path = "yolov8s.pt"

    # Pre-download base weights if missing
    if not os.path.exists(weights_path):
        print("⬇️ Downloading base model weights (yolov8s.pt)...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
        urllib.request.urlretrieve(url, weights_path)

    # Initialize and train
    model = YOLO(weights_path)
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=800,
        batch=16,
        patience=20,
        degrees=15.0,
        fliplr=0.5,
        mosaic=1.0,
        project="leaf_disease_run",
        name="train_v2",
        exist_ok=True
    )
    print("✅ Training complete!")

if __name__ == "__main__":
    train()