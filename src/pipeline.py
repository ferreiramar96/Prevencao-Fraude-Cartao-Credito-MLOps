from src.data_processing import (
    load_data, standardize_features,
    apply_scalers, split_train_validation)
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("fraude-cartao-credito")

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

# ────────────────────────────────────────────────────────
# Função para registrar o modelo no MLFlow
# ────────────────────────────────────────────────────────
def handle_model_registry(model_name, metrics, run_id, client, threshold=0.8):
    
    if metrics['f1_score'] < threshold:
        print(f"{model_name} não atingiu threshold → apenas tracked")
        return
    
    model_uri = f"runs:/{run_id}/model"
    
    result = mlflow.register_model(model_uri, model_name)
    new_version = result.version

    print(f"Modelo {model_name} registrado: v{new_version}")

    try:
        champion = client.get_model_version_by_alias(name=model_name, alias="champion")
        champion_run_id = champion.run_id
        champion_metric = client.get_run(champion_run_id).data.metrics["f1_score"]

        if metrics['f1_score'] > champion_metric:
            print("Novo modelo é melhor 🚀 → virando champion")
            client.set_registered_model_alias(name=model_name, alias="champion", version=new_version)
        else:
            print("Modelo Atual → vira challenger")
            client.set_registered_model_alias(name=model_name, alias="challenger", version=new_version)

    except Exception:
        print("Nenhum champion → criando primeiro champion")
        client.set_registered_model_alias(name=model_name, alias="champion", version=new_version)

# ────────────────────────────────────────────────────────
# Executará todo o pipeline
# ────────────────────────────────────────────────────────
def run_training_pipeline():
    # Importa os Dados
    url_dados = "https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1"
    df = load_data(url_dados)
    
    # 1. Separando os dados de treino e validação (ainda brutos)
    x_train, x_val, y_train, y_val = split_train_validation(df)

    # 2. Padroniza os dados (Aprende apenas com o X_TRAIN para evitar leakage)
    x_train, scaler_amount, scaler_time = standardize_features(x_train)
    
    # 3. Aplica a padronização na Validação usando os scalers do treino
    x_val = apply_scalers(x_val, scaler_amount, scaler_time)

    # 4. Treinamento, Avaliação e Registry
    for name, config in models_config.items():
        
        with mlflow.start_run():
            model = config['model_class'](**config['params'])
            trained_model = train_model(model, x_train, y_train)
            metrics = evaluate_model(trained_model, x_val, y_val)

            mlflow.log_metrics(metrics)
            mlflow.log_params(config['params'])
            mlflow.sklearn.log_model(trained_model, 'model')
            mlflow.log_artifacts("data/processed", artifact_path="processed_data")

            run_id = mlflow.active_run().info.run_id
            client = MlflowClient()

            handle_model_registry(
                model_name=name,
                metrics=metrics,
                run_id=run_id,
                client=client
            )

if __name__ == "__main__":
    run_training_pipeline()