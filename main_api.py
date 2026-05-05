from src.predict import load_best_model
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd

# Dicionário global para guardar o modelo na memória
model_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Roda quando a API inicia
    print("🚀 Carregando melhor modelo do MLflow...")
    model_assets["best_model"] = load_best_model()
    print("✅ Pipeline carregado e pronto!")
    yield
    model_assets.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API está viva!"}

@app.post("/predict")
async def predict(dados: dict):
    model_data = model_assets.get("best_model")
    if not model_data:
        return {"error": "Modelo não carregado"}

    # Criamos o DataFrame com as colunas originais
    df = pd.DataFrame([dados])

    pipeline = model_data["model"]
    
    # O predict roda o pré-processamento automaticamente!
    prediction = pipeline.predict(df)
    proba = pipeline.predict_proba(df)

    return {
        "predict": prediction[0].item(),
        "proba_fraude": proba[0][1].item(),
        "model_info": {
            "name": model_data["model_name"],
            "f1_score": model_data["f1_score"]
        }
    }
