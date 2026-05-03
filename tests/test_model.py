import numpy as np
from sklearn.linear_model import LogisticRegression
from src.model_training import train_model
from src.model_evaluation import evaluate_model


# Dados sintéticos simples para os testes (evita download + MLflow)
def make_fake_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.array([0] * 50 + [1] * 50)      # 50 normais, 50 fraudes
    return X, y


def test_train_model_returns_fitted():
    X, y = make_fake_data()
    model = LogisticRegression(max_iter=200)
    trained = train_model(model, X, y)

    # Verifica que retornou um modelo treinado (tem predict)
    assert hasattr(trained, "predict")
    preds = trained.predict(X)
    assert len(preds) == len(y)


def test_evaluate_model_returns_metrics():
    X, y = make_fake_data()
    model = LogisticRegression(max_iter=200)
    trained = train_model(model, X, y)

    metrics = evaluate_model(trained, X, y)

    expected_keys = ["f1_score", "accuracy", "recall", "precision", "auc_roc"]
    for key in expected_keys:
        assert key in metrics, f"Métrica '{key}' não encontrada"
        assert isinstance(metrics[key], float), f"Métrica '{key}' não é float"
        assert 0.0 <= metrics[key] <= 1.0, f"Métrica '{key}' fora do intervalo [0,1]"
