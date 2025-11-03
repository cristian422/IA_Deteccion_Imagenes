from fastapi import FastAPI

app = FastAPI(title="IA Deteccion de alimentos",
    description="API para detectar alimentos utilizando modelos de inteligencia artificial.",
    version="1.0.0",
    
)

@app.get("/")
def read_root():
    return {"mensaje": "¡Hola, FastAPI está funcionando!"}

@app.get("/saludo/{nombre}")
def read_item(nombre: str):
    return {"saludo": f"Hola, {nombre}"}
