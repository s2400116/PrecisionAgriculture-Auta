import sys
import os
from ultralytics import YOLO

def predict(image_path, weights_path="best.pt"):
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        return

    model = YOLO(weights_path)
    
    results = model(image_path, imgsz=800)
    
    results[0].save(filename="prediction_output.jpg")
    print("Prediction saved to prediction_output.jpg")
    
    try:
        results[0].show()
    except Exception:
        pass

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    predict(img_path)