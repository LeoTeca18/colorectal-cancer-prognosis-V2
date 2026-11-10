"""Exporta apenas o modelo vencedor e as melhores métricas extraídas do Notebook
para a pasta `models/` consumida pela interface clínica Streamlit (`app.py`).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"


def export_all() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    p2_path = ARTIFACTS_DIR / "processed_splits_v2.pkl"
    p3_path = ARTIFACTS_DIR / "phase3_results_v2.pkl"
    p4_path = ARTIFACTS_DIR / "phase4_results_v2.pkl"

    with open(p2_path, "rb") as f:
        p2 = pickle.load(f)
    with open(p3_path, "rb") as f:
        p3 = pickle.load(f)
    with open(p4_path, "rb") as f:
        p4 = pickle.load(f)

    X_train: pd.DataFrame = p2["X_train"]
    feature_names = list(X_train.columns)
    cph_winner = p4["cph"]

    # 1. Exportar exclusivamente o modelo de sobrevivência vencedor (CoxPH)
    joblib.dump(cph_winner, MODELS_DIR / "cox_survival_model.pkl")

    # 2. Exportar coorte de referência real do Treino (1.000 doentes)
    ref_sample = X_train[feature_names].sample(
        n=min(1000, len(X_train)), random_state=42
    ).reset_index(drop=True)
    ref_sample.to_csv(MODELS_DIR / "reference_cohort.csv", index=False)

    # 3. Exportar apenas o classificador vencedor e as métricas oficiais para exibição
    bundle = {
        "best_model_name": p3["best_model_name"],
        "best_clf": p3["best_clf"],
        "platt_calibrator": p3["platt_calibrator"],
        "final_thr": float(p3["final_thr"]),
        "final_metrics": p3["final_metrics"],
        "test_cindex_iso": float(p4["test_cindex"]),
        "sens_12m": float(p4["sens_12m_clf"]),
        "spec_12m": float(p4["spec_12m_clf"]),
        "brier_12m": float(p4["brier_12m_cox"]),
    }
    joblib.dump(bundle, MODELS_DIR / "clinical_pipeline_bundle.pkl")

    # Remover modelo secundário se existir
    stacked_file = MODELS_DIR / "stacked_cox_model.pkl"
    if stacked_file.exists():
        stacked_file.unlink()

    print(f"[SUCESSO] Modelo vencedor e métricas sincronizados em `{MODELS_DIR}`.")


if __name__ == "__main__":
    export_all()
