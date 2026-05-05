from mlflow import MlflowClient
from mlflow.exceptions import RestException
import mlflow
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega o .env a partir do diretório onde este arquivo está (src/)
load_dotenv(Path(__file__).parent / ".env")

url_mlflow = os.getenv("URL_MLFLOW")
client = MlflowClient(tracking_uri=url_mlflow)
mlflow.set_tracking_uri(url_mlflow)
METRIC_NAME = 'f1_score'

def get_best_model():
    candidatos = []

    # lista todos os modelos registrados
    for registered_model in client.search_registered_models():
        model_name = registered_model.name

        # lista versões do modelo
        for version in client.search_model_versions(f"name='{model_name}'"):
            run_id = version.run_id

            if not run_id:
                continue

            try:
                run = client.get_run(run_id)
            except RestException:
                print(f"⚠️  Run {run_id} não encontrado, pulando versão {version.version} do modelo {model_name}")
                continue

            metric_value = run.data.metrics.get(METRIC_NAME)

            if metric_value is None:
                continue

            candidatos.append({
                "model_name": model_name,
                "version": version.version,
                "run_id": run_id,
                "metric": metric_value
            })

    # pega o melhor
    if not candidatos:
        print("❌ Nenhum modelo com métrica válida encontrado")
        return None

    melhor = max(candidatos, key=lambda x: x["metric"])

    print(f"Melhor modelo: {melhor}")
    return melhor

def load_best_model():
    melhor = get_best_model()

    if melhor == None:
        return None

    model_name = melhor['model_name']
    model_version = melhor['version']
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    print("Modelo Carregado com Sucesso!")
    return {"model": model,
            "model_name": model_name,
            "f1_score": melhor["metric"]}