# api.py
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = (128, 128)
RUTA_MODELO = "export/modelo_FrutasLacteosGarbanzos.h5"
RUTA_LABELS = "export/labels.txt"

# Cargar modelo y labels una sola vez
modelo = load_model(RUTA_MODELO)
with open(RUTA_LABELS, "r", encoding="utf-8") as f:
    NOMBRES_CLASES = [line.strip() for line in f.readlines()]

app = FastAPI()

# CORS para permitir peticiones desde tu frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # en prod pon aquí tu dominio React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predecir_imagen_bytes(file_bytes: bytes):
    # Abrir imagen desde bytes
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probs = modelo.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    clase = NOMBRES_CLASES[idx]
    prob = float(probs[idx])

    return {
        "clase": clase,
        "indice": idx,
        "probabilidad": prob,
        "vector_probabilidades": probs.tolist()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    resultado = predecir_imagen_bytes(file_bytes)
    return resultado
