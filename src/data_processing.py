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
# 2. Separação do conjunto de teste holdout
# ────────────────────────────────────────────────────────
def split_holdout_test(df:pd.DataFrame):
    """Separa uma fração dos dados como conjunto de teste holdout.

    Essa separação é feita antes de qualquer transformação para
    simular dados completamente novos na avaliação final.

    Args:
        df: DataFrame bruto completo.

    Returns:
        Tupla (df_train, df_test) com os dados de treino e teste.
    """
    df_test = df.sample(frac=0.15, random_state=0)
    df_train = df.drop(df_test.index)
    return df_train, df_test


# ────────────────────────────────────────────────────────
# 3. Padronização das variáveis Amount e Time
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

    df.drop(columns=["Time", "Amount"], axis=1, inplace=True)

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

    df.drop(columns=["Time", "Amount"], axis=1, inplace=True)

    return df


# ────────────────────────────────────────────────────────
# 4. Separação X / y e Train / Validation split
# ────────────────────────────────────────────────────────
def split_train_validation(df:pd.DataFrame, random_state:int=None):
    """Separa os dados em features (X) e target (y), dividindo em treino/validação.

    Args:
        df: DataFrame padronizado.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (X_train, X_val, y_train, y_val).
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.25,random_state=random_state)

    return X_train, X_val, y_train, y_val


# ────────────────────────────────────────────────────────
# 5. Balanceamento de classes (UnderSampling)
# ────────────────────────────────────────────────────────
def balance_classes(X_train, y_train, random_state:int=None):
    """Aplica RandomUnderSampler para balancear as classes de treino.

    O dataset de fraude é altamente desbalanceado (~0.17% fraudes).
    O undersampling reduz a classe majoritária para igualar a minoritária.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (X_balanced, y_balanced).
    """
    under_sampler = RandomUnderSampler(random_state=random_state)
    X_balanced, y_balanced = under_sampler.fit_resample(X_train, y_train)
    return X_balanced, y_balanced


# ────────────────────────────────────────────────────────
# 6. Preparação do conjunto de teste holdout
# ────────────────────────────────────────────────────────
def prepare_test_data(df_test, scaler_amount, scaler_time):
    """Prepara o conjunto de teste holdout para avaliação final.

    Aplica os scalers já ajustados no treino e separa em X_test / y_test.

    Args:
        df_test: DataFrame de teste holdout (bruto).
        scaler_amount: StandardScaler ajustado na coluna Amount do treino.
        scaler_time: StandardScaler ajustado na coluna Time do treino.
        target_col: Nome da coluna-alvo.

    Returns:
        Tupla (X_test, y_test).
    """
    df_test_std = apply_scalers(df_test, scaler_amount, scaler_time)
    X_test = df_test_std.drop(columns="Class", axis=1)
    y_test = df_test_std["Class"]
    return X_test, y_test


# ────────────────────────────────────────────────────────
# 7. Exportação dos dados processados
# ────────────────────────────────────────────────────────
def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, output_dir="../data/processed"):
    """Salva os dados processados em arquivos CSV.

    Arquivos gerados:
        - X_train.csv, y_train.csv
        - X_val.csv, y_val.csv
        - X_test.csv, y_test.csv

    Args:
        X_train: Features de treino (balanceadas).
        y_train: Target de treino (balanceado).
        X_val: Features de validação.
        y_val: Target de validação.
        X_test: Features de teste holdout.
        y_test: Target de teste holdout.
        output_dir: Diretório de saída.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)