import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import numpy as np
import os
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr

# ==============================================================================
# ML Model Definition
# ==============================================================================

class FetusUltrasoundModel:
    def __init__(self, model_path="fetus_model.pth", num_classes=2, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.num_classes = num_classes
        
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        
        self._load_or_initialize_weights()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ["Healthy Fetus", "Not Healthy (Abnormal)"]

    def _load_or_initialize_weights(self):
        if os.path.exists(self.model_path):
            print(f"[*] Loading production model weights from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print(f"[!] Model file {self.model_path} not found. Creating a baseline model file...")
            torch.save(self.model.state_dict(), self.model_path)
            print(f"[*] Baseline model saved to {self.model_path}.")

    def preprocess_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
        processed_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return processed_bgr

    @torch.no_grad()
    def predict(self, frame):
        preprocessed_frame = self.preprocess_opencv(frame)
        input_tensor = self.transform(preprocessed_frame).unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
        return self.class_names[predicted_class.item()], confidence.item(), preprocessed_frame

# ==============================================================================
# FastAPI & REST API Endpoints
# ==============================================================================
app = FastAPI(title="Fetus Ultrasound ML API", description="Production API for Fetus Monitoring")
monitor = FetusUltrasoundModel()

@app.get("/")
def health_check():
    """GET Request: Checks if the API is running."""
    return {"status": "API is live!", "model": "MobileNetV3 (Healthy vs Not Healthy)"}

@app.post("/predict")
async def api_predict(file: UploadFile = File(...)):
    """POST Request: Send an image here from another website to get predictions."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format."})
        
    label, conf, preprocessed_view = monitor.predict(frame)
    
    # Convert OpenCV image to Base64 so other websites can display the noise-reduced image
    _, buffer = cv2.imencode('.jpg', preprocessed_view)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "prediction": label,
        "confidence": round(float(conf) * 100, 2),
        "preprocessed_image_base64": img_base64
    }

# ==============================================================================
# Gradio UI (Mounted on FastAPI)
# ==============================================================================
def process_image_gradio(image):
    if image is None: return None, None, "No image"
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label, conf, preproc = monitor.predict(frame_bgr)
    result_rgb = image.copy()
    color = (0, 255, 0) if conf > 0.8 else (255, 165, 0)
    cv2.putText(result_rgb, f"{label} ({conf*100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return result_rgb, cv2.cvtColor(preproc, cv2.COLOR_BGR2RGB), f"**{label}**\nConf: {conf*100:.2f}%"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Fetus Ultrasound Local UI\n(REST API is active at `POST /predict`)")
    with gr.Row():
        with gr.Column():
            in_img = gr.Image(sources=["upload", "webcam"])
            btn = gr.Button("Analyze")
        with gr.Column():
            out_img = gr.Image(label="Result")
            out_preproc = gr.Image(label="Enhanced")
            out_txt = gr.Markdown()
    btn.click(fn=process_image_gradio, inputs=[in_img], outputs=[out_img, out_preproc, out_txt])

# Mount Gradio onto the FastAPI app so both UI and API run on the same server
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    print("[*] Starting Production Server (FastAPI + Gradio)...")
    uvicorn.run(app, host="127.0.0.1", port=7860)
