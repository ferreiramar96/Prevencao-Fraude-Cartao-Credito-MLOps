"""
Módulo de Processamento de Dados e Feature Engineering.

Contém funções para carregar, limpar, pré-processar e transformar
os dados do dataset de detecção de fraude em cartões de crédito.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


# ────────────────────────────────────────────────────────
# 1. Carga dos dados
# ────────────────────────────────────────────────────────
def load_data(filepath:str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV.

    Args:
        filepath: Caminho (local ou URL) para o arquivo CSV.

    Returns:
        DataFrame com os dados brutos.
    """
    df = pd.read_csv(filepath)
    return df

# ────────────────────────────────────────────────────────
# 2. Padronização das variáveis Amount e Time
# ────────────────────────────────────────────────────────
def standardize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, StandardScaler]:
    """Padroniza as colunas 'Amount' e 'Time' usando StandardScaler.

    Cria as colunas 'std_amount' e 'std_time' e remove as originais.

    Args:
        df: DataFrame com as colunas Amount e Time.

    Returns:
        Tupla (df_padronizado, scaler_amount, scaler_time).
        Os scalers são retornados para reutilização no conjunto de teste.
    """
    df = df.copy()

    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    df["std_amount"] = scaler_amount.fit_transform(df["Amount"].values.reshape(-1, 1)) 
    df["std_time"] = scaler_time.fit_transform(df["Time"].values.reshape(-1, 1))

    df.drop(columns=["Time", "Amount"], inplace=True)

    return df, scaler_amount, scaler_time


def apply_scalers(df:pd.DataFrame, scaler_amount:StandardScaler, scaler_time:StandardScaler) -> pd.DataFrame:
    """Aplica scalers já ajustados ao conjunto de teste.

    Args:
        df: DataFrame de teste com colunas Amount e Time.
        scaler_amount: Scaler ajustado na coluna Amount do treino.
        scaler_time: Scaler ajustado na coluna Time do treino.

    Returns:
        DataFrame padronizado (sem as colunas originais Amount e Time).
    """
    df = df.copy()

    df["std_amount"] = scaler_amount.transform(df["Amount"].values.reshape(-1, 1))
    df["std_time"] = scaler_time.transform(df["Time"].values.reshape(-1, 1))

    df.drop(columns=["Time", "Amount"], inplace=True)

    return df


# ────────────────────────────────────────────────────────
# 3. Separação X / y e Train / Validation split
# ────────────────────────────────────────────────────────
def split_train_validation(df:pd.DataFrame):
    """Separa os dados em features (X) e target (y), dividindo em treino/validação.

    Args:
        df: DataFrame padronizado.

    Returns:
        Tupla (X_train, X_val, y_train, y_val).
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val


# ────────────────────────────────────────────────────────
# 4. Balanceamento de classes (UnderSampling)
# ────────────────────────────────────────────────────────
def balance_classes(X_train, y_train):
    """Aplica RandomUnderSampler para balancear as classes de treino.

    O dataset de fraude é altamente desbalanceado
    O undersampling reduz a classe majoritária para igualar a minoritária.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.

    Returns:
        Tupla (X_balanced, y_balanced).
    """
    under_sampler = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = under_sampler.fit_resample(X_train, y_train)
    return X_balanced, y_balanced