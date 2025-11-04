# POC – Predição de Vendas com TensorFlow

Este pacote contém:
- `sales_dataset.csv` — base sintética diária com colunas: `date`, `value`, `num_items`.
- `train_sales_predictor.py` — script que treina um modelo Keras para prever o **valor de vendas do dia seguinte**.
- `model_meta.json` — será gerado após o treino, com metadados do modelo.
- `sales_value_model.keras` — será gerado após o treino.

## Como usar

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -U tensorflow pandas numpy scikit-learn

python train_sales_predictor.py --csv_path sales_dataset.csv --out_model sales_value_model.keras --lags 7
```

Ao final, o script imprime a **MAE** no conjunto de teste e uma previsão para o dia seguinte ao último dia da base.

## Estrutura de Features
- Sinais calendários: seno/cosseno de dia da semana e mês; índice de tempo (tendência).
- Lags: `value_lag_1..7` e `items_lag_1..7`.
- Target: `value` do **dia seguinte**.

A base foi gerada com sazonalidade semanal/mensal, tendência leve e ruídos (promoções/eventos).
