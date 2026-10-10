import cv2
import numpy as np
import os
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr
import onnxruntime as ort

# ==============================================================================
# ML Model Definition (ONNX Runtime)
# ==============================================================================

class FetusUltrasoundModel:
    def __init__(self, model_path="fetus_model.onnx", num_classes=2):
        self.model_path = model_path
        self.class_names = ["Healthy Fetus", "Not Healthy (Abnormal)"]
        
        if not os.path.exists(self.model_path):
            print(f"[!] Warning: {self.model_path} not found. Ensure you uploaded the ONNX model.")
            self.session = None
        else:
            self.session = ort.InferenceSession(self.model_path)
            print(f"[*] Loaded highly optimized ONNX model from {self.model_path}")
            
    def preprocess_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
        processed_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return processed_bgr

    def _transform(self, image):
        # Resize to 224x224
        img = cv2.resize(image, (224, 224))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, frame):
        preprocessed_frame = self.preprocess_opencv(frame)
        if self.session is None:
            return "Model Not Found", 0.0, preprocessed_frame
            
        input_tensor = self._transform(preprocessed_frame)
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        # Run ONNX inference
        outputs = self.session.run([output_name], {input_name: input_tensor})[0]
        
        # Softmax
        exp_preds = np.exp(outputs[0] - np.max(outputs[0]))
        probabilities = exp_preds / np.sum(exp_preds)
        
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return self.class_names[predicted_class], float(confidence), preprocessed_frame

# ==============================================================================
# FastAPI & REST API Endpoints
# ==============================================================================
app = FastAPI(title="Fetus Ultrasound ML API", description="Production API for Fetus Monitoring")
monitor = FetusUltrasoundModel()

@app.get("/")
def health_check():
    """GET Request: Checks if the API is running."""
    return {"status": "API is live!", "model": "ONNX MobileNetV3 (Healthy vs Not Healthy)"}

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

with gr.Blocks() as demo:
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
    if not os.environ.get("SPACE_ID"):
        print("[*] Starting Production Server (FastAPI + Gradio) locally...")
        uvicorn.run(app, host="0.0.0.0", port=7860)
