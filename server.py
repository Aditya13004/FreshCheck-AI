import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="FreshCheck AI API")

# Allow Frontend to communicate with Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Deep Learning Model...")
model = keras.models.load_model("best_model.keras")

# Dynamically build classes from dataset
fruit_set = set()
if os.path.exists("dataset/train"):
    for root, dirs, files in os.walk("dataset/train"):
        folder = os.path.basename(root).lower()
        if folder.startswith('fresh'):
            fruit_set.add(folder.replace('fresh', ''))
        elif folder.startswith('rotten'):
            fruit_set.add(folder.replace('rotten', ''))
            
fruit_to_idx = {fruit: idx for idx, fruit in enumerate(sorted(list(fruit_set)))}
idx_to_fruit = {str(v): k for k, v in fruit_to_idx.items()}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    
    # Handle PNG transparent channels
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Run Inference
    fruit_preds, cond_preds = model.predict(img_array)
    
    # Process Fruit Type
    fruit_idx = np.argmax(fruit_preds[0])
    fruit_name = idx_to_fruit[str(fruit_idx)].capitalize()
    fruit_confidence = float(fruit_preds[0][fruit_idx]) * 100
    
    # Out-Of-Distribution (OOD) Filter
    # Neural Networks are inherently overconfident. We set a strict threshold to filter non-produce.
    if fruit_confidence < 98.5:
        fruit_name = "Unknown Object"
        condition = "N/A"
        cond_confidence = 0.0
    else:
        # Process Condition
        cond_score = float(cond_preds[0][0])
        condition = "Rotten" if cond_score > 0.5 else "Fresh"
        cond_confidence = cond_score * 100 if condition == "Rotten" else (1 - cond_score) * 100
        
    # --- EXPLAINABLE AI: Activation Heatmap ---
    heatmap_b64 = None
    if fruit_name != "Unknown Object":
        try:
            import cv2
            import base64
            # Extract activations from the MobileNetV2 backbone
            backbone = [layer for layer in model.layers if isinstance(layer, keras.Model)][0]
            activation_model = keras.models.Model(inputs=backbone.input, outputs=backbone.output)
            activations = activation_model.predict(img_array)
            
            # Generate Class-Agnostic Activation Map
            hm = np.mean(activations, axis=3)[0]
            hm = np.maximum(hm, 0)
            hm /= np.max(hm)
            
            # Superimpose onto original image
            hm = cv2.resize(hm, (img_resized.width, img_resized.height))
            hm = np.uint8(255 * hm)
            hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            
            original_img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
            superimposed_img = cv2.addWeighted(original_img_bgr, 0.6, hm_color, 0.4, 0)
            
            _, buffer = cv2.imencode('.jpg', superimposed_img)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print("Heatmap generation failed:", e)
    
    return {
        "fruit": fruit_name,
        "fruit_confidence": fruit_confidence,
        "condition": condition,
        "cond_confidence": cond_confidence,
        "heatmap": heatmap_b64
    }

# Serve the beautiful frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting FreshCheck AI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
