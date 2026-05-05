"""
Módulo de Processamento de Dados e Feature Engineering.

Contém funções para carregar, separar e preparar o pré-processamento
dos dados do dataset de detecção de fraude em cartões de crédito.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler

# ────────────────────────────────────────────────────────
# 1. Carga dos dados
# ────────────────────────────────────────────────────────
def load_data(filepath:str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV."""
    df = pd.read_csv(filepath)
    return df

# ────────────────────────────────────────────────────────
# 2. Criação do Pré-processador (Pipeline Stage)
# ────────────────────────────────────────────────────────
def get_preprocessor():
    """Cria um ColumnTransformer para padronizar Amount e Time.
    
    Returns:
        Um objeto ColumnTransformer pronto para ser usado em um Pipeline.
    """
    numeric_features = ['Amount', 'Time']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

# ────────────────────────────────────────────────────────
# 3. Separação X / y e Train / Validation split
# ────────────────────────────────────────────────────────
def split_train_validation(df:pd.DataFrame):
    """Separa os dados em features (X) e target (y), dividindo em treino/validação."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

# ────────────────────────────────────────────────────────
# 4. Balanceamento de classes (UnderSampling)
# ────────────────────────────────────────────────────────
def balance_classes(X_train, y_train):
    """Aplica RandomUnderSampler para balancear as classes de treino."""
    under_sampler = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = under_sampler.fit_resample(X_train, y_train)
    return X_balanced, y_balanced