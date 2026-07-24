
import sys
from ultralytics import YOLO

def predict(image_path, weights_path="best.pt"):
    # Load trained model
    model = YOLO(weights_path)
    
    # Perform inference
    results = model(image_path)
    
    # Display bounding box output
    results[0].show()
    
    # Save output image
    results[0].save(filename="prediction_output.jpg")
    print("✅ Prediction saved to prediction_output.jpg")

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    predict(img_path)
