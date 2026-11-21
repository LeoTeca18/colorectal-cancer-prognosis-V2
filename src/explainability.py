"""Explicabilidade local SHAP e LIME (Pilar C).

Alinhado integralmente com o protocolo do Notebook:
1. SHAP (`TreeExplainer`) sobre o classificador campeão `LightGBM (Otimizado)` nas 43 features.
2. SHAP (`Explainer` independente) e LIME Tabular sobre o `log partial hazard` do modelo `CoxPH`
   nas exatas 26 covariáveis independentes decodificadas sem perdas de `reference_cohort.csv`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .data_validation import format_category_pt
from .inference import (
    OHE_SPEC,
    RAW_FEATURE_COLUMNS,
    decode_ohe_to_ui,
    encode_dataframe,
    encode_patient,
    load_reference_cohort,
)

EXP_SCHEMA_VERSION = "raw_26_pt_v5"

FRIENDLY_NAMES: Dict[str, str] = {
    "age": "Idade (anos)",
    "bmi": "IMC (kg/m²)",
    "gender": "Sexo Biológico",
    "race": "Grupo Étnico/Racial",
    "region": "Região",
    "urban_or_rural": "Zona de Residência",
    "socioeconomic_status": "Nível Socioeconómico",
    "family_history": "História Familiar de CCR",
    "previous_cancer_history": "História Prévia de Cancro",
    "stage_at_diagnosis": "Estádio ao Diagnóstico",
    "tumor_aggressiveness": "Agressividade Tumoral",
    "colonoscopy_access": "Acesso a Colonoscopia",
    "screening_regularity": "Regularidade de Rastreio",
    "diet_type": "Padrão Alimentar",
    "physical_activity_level": "Atividade Física",
    "smoking_status": "Tabagismo",
    "alcohol_consumption": "Consumo de Álcool",
    "red_meat_consumption": "Carne Vermelha",
    "fiber_consumption": "Consumo de Fibra",
    "insurance_coverage": "Seguro de Saúde",
    "time_to_diagnosis": "Tempo até Diagnóstico",
    "treatment_access": "Acesso ao Tratamento",
    "chemotherapy_received": "Quimioterapia",
    "radiotherapy_received": "Radioterapia",
    "surgery_received": "Cirurgia",
    "follow_up_adherence": "Adesão ao Acompanhamento",
}

OHE_FEATURE_LABELS_PT: List[str] = ["Idade (anos)", "IMC (kg/m²)"] + [
    f"{FRIENDLY_NAMES[field]}: {format_category_pt(field, cat)}"
    for field, _, _, cats in OHE_SPEC
    for cat in cats
]

# Categorias ordenadas por variável categórica (índice 0 = categoria de referência omitida no OHE)
CATEGORY_ORDER: Dict[str, List[str]] = {
    field: [ref_cat, *cats] for field, _, ref_cat, cats in OHE_SPEC
}


def _raw_df_to_numeric_matrix(df_raw: pd.DataFrame) -> np.ndarray:
    """Converte as 26 colunas brutas em matriz numérica (N, 26) com códigos inteiros para categóricas."""
    cols: List[np.ndarray] = [
        df_raw["age"].to_numpy(dtype=float),
        df_raw["bmi"].to_numpy(dtype=float),
    ]
    for field in RAW_FEATURE_COLUMNS[2:]:
        cats = CATEGORY_ORDER[field]
        mapping = {c: float(idx) for idx, c in enumerate(cats)}
        cols.append(df_raw[field].astype(str).map(mapping).fillna(0.0).to_numpy(dtype=float))
    return np.column_stack(cols)


def _numeric_matrix_to_model_df(X_num: np.ndarray) -> pd.DataFrame:
    """Converte matriz numérica (N, 26) de volta para as 43 colunas One-Hot do modelo."""
    X_arr = np.atleast_2d(np.asarray(X_num, dtype=float))
    raw_dict: Dict[str, Any] = {
        "age": X_arr[:, 0],
        "bmi": X_arr[:, 1],
    }
    for col_idx, field in enumerate(RAW_FEATURE_COLUMNS[2:], start=2):
        cats = CATEGORY_ORDER[field]
        codes = np.clip(np.rint(X_arr[:, col_idx]).astype(int), 0, len(cats) - 1)
        raw_dict[field] = np.asarray(cats, dtype=object)[codes]
    return encode_dataframe(pd.DataFrame(raw_dict, columns=RAW_FEATURE_COLUMNS))


def encode_ui_patient(patient_data: Dict[str, Any]) -> np.ndarray:
    """Extrai o vetor numérico (1, 26) correspondente às 26 covariáveis da interface."""
    return _raw_df_to_numeric_matrix(pd.DataFrame([patient_data], columns=RAW_FEATURE_COLUMNS))


def _log_hazard_ui(model: Any):
    def f(X_num: np.ndarray) -> np.ndarray:
        df_model = _numeric_matrix_to_model_df(X_num)
        return np.asarray(model.predict_log_partial_hazard(df_model)).ravel()

    return f


# ============================================================================
# 1. SHAP TreeExplainer — LightGBM Campeão (alinhado com Fase 5.4 do Notebook)
# ============================================================================
def build_lgb_tree_explainer(bundle: Dict[str, Any]) -> shap.TreeExplainer:
    """Cria `shap.TreeExplainer` idêntico ao da Fase 5.4 sobre o LightGBM campeão."""
    return shap.TreeExplainer(bundle["best_clf"])


def generate_lgb_shap_plot(
    patient_data: Dict[str, Any],
    bundle: Dict[str, Any],
    explainer: Optional[shap.TreeExplainer] = None,
    max_display: int = 15,
):
    """SHAP Waterfall local do LightGBM campeão (classe 1 — Recidiva) nas 43 features OHE."""
    if explainer is None:
        explainer = build_lgb_tree_explainer(bundle)

    X_model = encode_patient(patient_data)
    sv = explainer.shap_values(X_model)
    ev = explainer.expected_value

    if isinstance(sv, list):
        sv_pos = np.asarray(sv[1])[0]
        base_val = float(ev[1]) if isinstance(ev, (list, tuple, np.ndarray)) else float(ev)
    elif np.asarray(sv).ndim == 3:
        sv_pos = np.asarray(sv)[0, :, 1]
        base_val = float(np.asarray(ev).ravel()[-1])
    else:
        sv_pos = np.asarray(sv)[0]
        base_val = float(np.asarray(ev).ravel()[0])

    exp = shap.Explanation(
        values=sv_pos,
        base_values=base_val,
        data=X_model.iloc[0].to_numpy(dtype=float),
        feature_names=OHE_FEATURE_LABELS_PT,
    )
    plt.close("all")
    fig = plt.figure(figsize=(8.8, 6.8))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.8, 6.8)
    fig.tight_layout()
    return fig


# ============================================================================
# 2. SHAP & LIME — CoxPH Isolado nas 26 covariáveis independentes
# ============================================================================
def build_explainer(model: Any, background: Optional[pd.DataFrame] = None) -> shap.Explainer:
    """Cria o explicador SHAP para o CoxPH nas 26 variáveis decodificadas de `reference_cohort.csv`."""
    if background is None:
        background = load_reference_cohort(model, n=100)
    raw_bg = decode_ohe_to_ui(background)
    bg_num = _raw_df_to_numeric_matrix(raw_bg)
    masker = shap.maskers.Independent(bg_num, max_samples=min(100, len(bg_num)))
    explainer = shap.Explainer(
        _log_hazard_ui(model),
        masker,
        feature_names=[FRIENDLY_NAMES[c] for c in RAW_FEATURE_COLUMNS],
    )
    setattr(explainer, "_ui_schema_version", EXP_SCHEMA_VERSION)
    return explainer


def generate_shap_plot(patient_data: Dict[str, Any], model: Any, explainer: Optional[shap.Explainer] = None):
    """Retorna Figura Matplotlib com SHAP Waterfall do CoxPH nas 26 covariáveis."""
    if explainer is None or getattr(explainer, "_ui_schema_version", None) != EXP_SCHEMA_VERSION:
        explainer = build_explainer(model)
    X_num = encode_ui_patient(patient_data)
    try:
        exp = explainer(X_num)
    except Exception:
        explainer = build_explainer(model)
        exp = explainer(X_num)
    plt.close("all")
    fig = plt.figure(figsize=(8.8, 7.2))
    shap.plots.waterfall(exp[0], max_display=16, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.8, 7.2)
    fig.tight_layout()
    return fig


def build_lime_explainer(model: Any, background: Optional[pd.DataFrame] = None):
    """Cria o explicador LIME Tabular nas 26 variáveis reais decodificadas de `reference_cohort.csv`."""
    from lime.lime_tabular import LimeTabularExplainer

    if background is None:
        background = load_reference_cohort(model, n=1000)
    raw_bg = decode_ohe_to_ui(background)
    bg_num = _raw_df_to_numeric_matrix(raw_bg)

    cat_idx = list(range(2, len(RAW_FEATURE_COLUMNS)))
    cat_names = {
        idx: [format_category_pt(field, c) for c in CATEGORY_ORDER[field]]
        for idx, field in enumerate(RAW_FEATURE_COLUMNS[2:], start=2)
    }

    lime_exp = LimeTabularExplainer(
        bg_num,
        feature_names=[FRIENDLY_NAMES[c] for c in RAW_FEATURE_COLUMNS],
        categorical_features=cat_idx,
        categorical_names=cat_names,
        mode="regression",
        discretize_continuous=True,
        random_state=0,
    )
    setattr(lime_exp, "_ui_schema_version", EXP_SCHEMA_VERSION)
    return lime_exp


def generate_lime_plot(
    patient_data: Dict[str, Any],
    model: Any,
    lime_explainer: Any = None,
    num_features: int = 12,
    num_samples: int = 2000,
):
    """Retorna `(Figure, DataFrame)` com a explicação LIME local do CoxPH nas 26 variáveis."""
    if lime_explainer is None or getattr(lime_explainer, "_ui_schema_version", None) != EXP_SCHEMA_VERSION:
        lime_explainer = build_lime_explainer(model)
    X_num = encode_ui_patient(patient_data)[0]
    try:
        exp = lime_explainer.explain_instance(
            X_num,
            _log_hazard_ui(model),
            num_features=num_features,
            num_samples=num_samples,
        )
    except Exception:
        lime_explainer = build_lime_explainer(model)
        exp = lime_explainer.explain_instance(
            X_num,
            _log_hazard_ui(model),
            num_features=num_features,
            num_samples=num_samples,
        )
    items = exp.as_list()
    df = pd.DataFrame(items, columns=["Condição", "Peso"])
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.8, 0.45 * len(df) + 1.4))
    colors = ["#f87171" if w > 0 else "#34d399" for w in df["Peso"]]
    ax.barh(df["Condição"][::-1], df["Peso"][::-1], color=colors[::-1])
    ax.axvline(0, color="#64748b", lw=1)
    ax.set_xlabel("Contribuição para o log-hazard (vermelho = ↑ risco)")
    ax.set_title(f"LIME (CoxPH) — fidelidade local R² = {exp.score:.2f}")
    fig.tight_layout()
    return fig, df
