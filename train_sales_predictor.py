#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treinador de modelo de predição de vendas com TensorFlow (Keras).

- Entrada: CSV com colunas ["date","value","num_items"], uma linha por dia.
- Saída: Modelo salvo (sales_value_model.keras) que prediz o valor de vendas do DIA SEGUINTE
         a partir de atributos de calendário e lags (últimos 7 dias).
         
Como executar:
    1) python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
       # No Windows (PowerShell):
       #   py -3 -m venv .venv ; .\.venv\Scripts\Activate.ps1
    2) pip install -U tensorflow pandas numpy scikit-learn
    3) python train_sales_predictor.py --csv_path sales_dataset.csv
    
Dica: Ajuste épocas/neurônios se quiser treinar mais/menos.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf

def build_features(df: pd.DataFrame, lags=7):
    """Gera features com sin/cos de calendário e lags de 'value' e 'num_items'.
       Retorna (X, y, feature_names, dates) onde y = value no dia seguinte.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Atributos de calendário
    df["dayofweek"] = df["date"].dt.dayofweek  # 0..6
    df["month"] = df["date"].dt.month          # 1..12
    df["day_sin"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["day_cos"] = np.cos(2*np.pi*df["dayofweek"]/7)
    df["mon_sin"] = np.sin(2*np.pi*(df["month"]-1)/12)
    df["mon_cos"] = np.cos(2*np.pi*(df["month"]-1)/12)
    # Índice de tempo normalizado (tendência)
    df["t_idx"] = (df.index - df.index.min())/(df.index.max() - df.index.min()+1e-9)
    
    # Lags
    for k in range(1, lags+1):
        df[f"value_lag_{k}"] = df["value"].shift(k)
        df[f"items_lag_{k}"] = df["num_items"].shift(k)
    
    # Target = valor do dia seguinte
    df["target_next_value"] = df["value"].shift(-1)
    
    # Remove linhas com NaN gerados por shift
    df2 = df.dropna().reset_index(drop=True)
    
    feature_cols = ["day_sin","day_cos","mon_sin","mon_cos","t_idx"]
    feature_cols += [f"value_lag_{k}" for k in range(1, lags+1)]
    feature_cols += [f"items_lag_{k}" for k in range(1, lags+1)]
    
    X = df2[feature_cols].astype("float32").values
    y = df2["target_next_value"].astype("float32").values
    dates = df2["date"].astype(str).tolist()
    return X, y, feature_cols, dates

def make_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    norm = tf.keras.layers.Normalization(name="norm")(inputs)  # adapt depois
    x = tf.keras.layers.Dense(64, activation="relu")(norm)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, name="pred_value")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model

def train(csv_path: str, out_model_path: str = "sales_value_model.keras", lags: int = 7):
    df = pd.read_csv(csv_path)
    X, y, feature_names, dates = build_features(df, lags=lags)
    
    # Split temporal: 80% treino / 20% teste
    n = len(X)
    n_train = int(n*0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test,  y_test  = X[n_train:], y[n_train:]
    
    model = make_model(X.shape[1])
    # Adapt Normalization no treino
    norm_layer = model.get_layer("norm")
    norm_layer.adapt(X_train)
    
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                          restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[es],
        verbose=2
    )
    
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[TEST] MSE={test_loss:.2f}  MAE={test_mae:.2f}")
    
    # Salvar modelo
    model.save(out_model_path)
    print(f"Modelo salvo em: {out_model_path}")
    
    # Persistir metadados
    meta = {
        "feature_names": feature_names,
        "lags": lags,
        "train_size": n_train,
        "total_rows": n,
        "test_metrics": {"mse": float(test_loss), "mae": float(test_mae)},
    }
    with open("model_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("Metadados salvos em: model_meta.json")
    
    # Exemplo de inferência para o dia seguinte ao último do dataset
    # (usa as últimas janelas disponíveis)
    # Preparar a última linha de features:
    # Reconstroi as features a partir do df original para facilitar.
    X_all, _, feature_names_all, dates_all = build_features(df, lags=lags)
    last_feat = X_all[-1:]
    next_value_pred = float(model.predict(last_feat, verbose=0).ravel()[0])
    print(f"Previsão do valor de vendas para o próximo dia após {dates_all[-1]}: R$ {next_value_pred:,.2f}")
    return out_model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="sales_dataset.csv",
                        help="Caminho para o CSV de dados de vendas")
    parser.add_argument("--out_model", type=str, default="sales_value_model.keras",
                        help="Caminho de saída do modelo (.keras)")
    parser.add_argument("--lags", type=int, default=7,
                        help="Quantidade de lags diários para usar como features")
    args = parser.parse_args()
    
    train(csv_path=args.csv_path, out_model_path=args.out_model, lags=args.lags)

if __name__ == "__main__":
    main()
