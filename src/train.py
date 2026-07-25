import os
from ultralytics import YOLO

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, "config", "data.yaml")

    model = YOLO("yolov8s.pt")
    
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
    print("Training complete!")

if __name__ == "__main__":
    train()