import gradio as gr
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO("best.pt")

def predict_leaf_disease(image):
    if image is None:
        return None

    results = model(image, imgsz=800)
    
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(annotated_rgb)

app = gr.Interface(
    fn=predict_leaf_disease,
    inputs=gr.Image(type="pil", label="Upload Crop/Leaf Image"),
    outputs=gr.Image(type="pil", label="Detection Results"),
    title="🌿 Precision Agriculture: Plant Leaf Anomaly Detection",
    description=(
        "Upload an image of a plant leaf to detect and localize agricultural anomalies. "
        "The model identifies **Fungal Infections**, **Pest Damage**, and general **Leaf Anomalies**."
    )
)

if __name__ == "__main__":
    app.launch(share=True, theme=gr.themes.Soft())