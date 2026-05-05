from src.data_processing import load_data, split_train_validation, get_preprocessor
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
import logging
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

# Suprir warnings internos do MLflow que não são acionáveis
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

url_mlflow = os.getenv("URL_MLFLOW")
mlflow.set_tracking_uri(url_mlflow)
mlflow.set_experiment("prevencao-fraude-cartao-credito")

models_config = {
    "LogisticRegression": {
        "model_class": LogisticRegression,
        "params": {
            "solver": "lbfgs",
            "max_iter": 100,
            "class_weight": "balanced"
        }
    },
    "DecisionTree": {
        "model_class": DecisionTreeClassifier,
        "params": {
            "class_weight": "balanced"
        }
    },
    "RandomForest": {
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": 50,
            "class_weight": "balanced"
        }
    }
}

def handle_model_registry(model_name, metrics, run_id, client, threshold=0.8):
    if metrics['f1_score'] < threshold:
        print(f"⚠️ {model_name} não atingiu o threshold ({metrics['f1_score']:.2f}) → Apenas Tracked.")
        return
    
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    new_version = result.version
    print(f"✅ Modelo {model_name} registrado: v{new_version}")

    try:
        champion = client.get_model_version_by_alias(name=model_name, alias="champion")
        champion_metric = client.get_run(champion.run_id).data.metrics["f1_score"]

        if metrics['f1_score'] > champion_metric:
            print(f"🚀 Novo modelo é melhor ({metrics['f1_score']:.4f} vs {champion_metric:.4f}) → Virando Champion!")
            client.set_registered_model_alias(name=model_name, alias="champion", version=new_version)
        else:
            print(f"📉 Modelo atual ({metrics['f1_score']:.4f}) é inferior ao Champion ({champion_metric:.4f}) → Challenger.")
            client.set_registered_model_alias(name=model_name, alias="challenger", version=new_version)
    except Exception:
        print(f"🥇 Nenhum Champion encontrado para {model_name} → Definindo v{new_version} como o primeiro Champion.")
        client.set_registered_model_alias(name=model_name, alias="champion", version=new_version)

def run_training_pipeline():
    print("📡 Carregando dados do Dropbox...")
    url_dados = "https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1"
    df = load_data(url_dados)
    
    x_train, x_val, y_train, y_val = split_train_validation(df)
    preprocessor = get_preprocessor()

    print(f"⚙️ Iniciando treinamento de {len(models_config)} modelos...")
    for name, config in models_config.items():
        print(f"\n--- Treinando: {name} ---")
        with mlflow.start_run():
            clf = config['model_class'](**config['params'])
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])

            pipeline.fit(x_train, y_train)
            metrics = evaluate_model(pipeline, x_val, y_val)

            mlflow.log_metrics(metrics)
            mlflow.log_params(config['params'])
            mlflow.sklearn.log_model(pipeline, name='model')
            
            run_id = mlflow.active_run().info.run_id
            client = MlflowClient()

            handle_model_registry(name, metrics, run_id, client)
    
    print("\n🏁 Pipeline de treinamento finalizado!")

if __name__ == "__main__":
    run_training_pipeline()