"""Explicabilidade local SHAP e LIME (Pilar C).

Opera exclusivamente sobre as 18 variáveis presentes na interface web (`app.py`),
com os mesmos nomes exatos dos campos da barra lateral.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .data_validation import WINSOR_BOUNDS
from .inference import FEATURE_COLUMNS, load_reference_cohort

EXP_SCHEMA_VERSION = "ui_18_v3"

# As exatas 18 variáveis que constam na barra lateral da interface (`app.py`)
UI_FEATURE_COLUMNS: List[str] = [
    "age",
    "sex_male",
    "bmi",
    "income",
    "t_stage",
    "n_stage",
    "m_stage",
    "grade",
    "family_history",
    "previous_cancer",
    "surgery",
    "chemo",
    "radiotherapy",
    "follow_up",
    "screening",
    "smoker",
    "alcohol",
    "activity",
]

# Nomes idênticos aos rótulos da barra lateral em `app.py`
FRIENDLY_NAMES: Dict[str, str] = {
    "age": "Idade (anos)",
    "sex_male": "Sexo Biológico",
    "bmi": "IMC (kg/m²)",
    "income": "Nível socioeconómico",
    "t_stage": "Estádio T",
    "n_stage": "Estádio N",
    "m_stage": "Estádio M",
    "grade": "Grau / agressividade tumoral",
    "family_history": "História familiar de CCR",
    "previous_cancer": "História prévia de cancro",
    "surgery": "Cirurgia realizada",
    "chemo": "Adesão à quimioterapia",
    "radiotherapy": "Radioterapia",
    "follow_up": "Adesão rigorosa ao follow-up",
    "screening": "Rastreio regular por colonoscopia",
    "smoker": "Fumador ativo",
    "alcohol": "Consumo elevado de álcool",
    "activity": "Atividade física regular",
}

BINARY_UI_FEATURES: List[str] = [
    "sex_male",
    "m_stage",
    "family_history",
    "previous_cancer",
    "surgery",
    "chemo",
    "radiotherapy",
    "follow_up",
    "screening",
    "smoker",
    "alcohol",
    "activity",
]


def encode_ui_patient(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """Extrai exclusivamente o vetor de 18 variáveis da interface a partir do dicionário do doente."""
    p = patient_data
    row = {
        "age": float(p["age"]),
        "sex_male": float(1 if p["sex"] == "M" else 0),
        "bmi": float(p["bmi"]),
        "income": float(p.get("income", 2)),
        "t_stage": float(p["t_stage"]),
        "n_stage": float(p["n_stage"]),
        "m_stage": float(p["m_stage"]),
        "grade": float(p["grade"]),
        "family_history": float(int(bool(p.get("family_history", 0)))),
        "previous_cancer": float(int(bool(p.get("previous_cancer", 0)))),
        "surgery": float(int(bool(p["surgery"]))),
        "chemo": float(int(bool(p["chemo"]))),
        "radiotherapy": float(int(bool(p["radiotherapy"]))),
        "follow_up": float(int(bool(p.get("follow_up", 1)))),
        "screening": float(int(bool(p.get("screening", 1)))),
        "smoker": float(int(bool(p["smoker"]))),
        "alcohol": float(int(bool(p["alcohol"]))),
        "activity": float(int(bool(p["activity"]))),
    }
    return pd.DataFrame([row], columns=UI_FEATURE_COLUMNS)


def _ui_matrix_to_model_df(X_ui: np.ndarray) -> pd.DataFrame:
    """Converte de forma vetorizada uma matriz `(N, 18)` das variáveis da UI para o DataFrame interno do modelo."""
    X_arr = np.atleast_2d(np.asarray(X_ui, dtype=float))
    n = X_arr.shape[0]
    ui_df = pd.DataFrame(X_arr, columns=UI_FEATURE_COLUMNS)

    age_lo, age_hi = WINSOR_BOUNDS["Age"]
    bmi_lo, bmi_hi = WINSOR_BOUNDS["BMI"]
    age = np.clip(ui_df["age"].to_numpy(), age_lo, age_hi)
    bmi = np.clip(ui_df["bmi"].to_numpy(), bmi_lo, bmi_hi)

    sex_male = (np.rint(ui_df["sex_male"].to_numpy()) == 1).astype(int)
    income = np.rint(ui_df["income"].to_numpy()).astype(int)
    t_stage = np.rint(ui_df["t_stage"].to_numpy()).astype(int)
    n_stage = np.rint(ui_df["n_stage"].to_numpy()).astype(int)
    m_stage = np.rint(ui_df["m_stage"].to_numpy()).astype(int)
    grade = np.rint(ui_df["grade"].to_numpy()).astype(int)

    family_history = (np.rint(ui_df["family_history"].to_numpy()) == 1).astype(int)
    previous_cancer = (np.rint(ui_df["previous_cancer"].to_numpy()) == 1).astype(int)
    surgery = (np.rint(ui_df["surgery"].to_numpy()) == 1).astype(int)
    chemo = (np.rint(ui_df["chemo"].to_numpy()) == 1).astype(int)
    radiotherapy = (np.rint(ui_df["radiotherapy"].to_numpy()) == 1).astype(int)
    follow_up = (np.rint(ui_df["follow_up"].to_numpy()) == 1).astype(int)
    screening = (np.rint(ui_df["screening"].to_numpy()) == 1).astype(int)
    smoker = (np.rint(ui_df["smoker"].to_numpy()) == 1).astype(int)
    alcohol = (np.rint(ui_df["alcohol"].to_numpy()) == 1).astype(int)
    activity = (np.rint(ui_df["activity"].to_numpy()) == 1).astype(int)

    # Derivação vetorizada de TNM -> Stage AJCC (I, II, III, IV)
    stage_iv = (m_stage == 1).astype(int)
    stage_iii = ((m_stage == 0) & (n_stage > 0)).astype(int)
    stage_ii = ((m_stage == 0) & (n_stage == 0) & (t_stage >= 3)).astype(int)

    zeros = np.zeros(n, dtype=int)
    ones = np.ones(n, dtype=int)

    data = {
        "Age": age,
        "BMI": bmi,
        "Gender_Male": sex_male,
        "Race_Black": zeros,
        "Race_Hispanic": zeros,
        "Race_Other": zeros,
        "Race_White": ones,
        "Region_Asia Pacific": zeros,
        "Region_Europe": ones,
        "Region_Latin America": zeros,
        "Region_North America": zeros,
        "Urban_or_Rural_Urban": ones,
        "Socioeconomic_Status_Low": (income <= 1).astype(int),
        "Socioeconomic_Status_Middle": (income == 2).astype(int),
        "Family_History_Yes": family_history,
        "Previous_Cancer_History_Yes": previous_cancer,
        "Stage_at_Diagnosis_II": stage_ii,
        "Stage_at_Diagnosis_III": stage_iii,
        "Stage_at_Diagnosis_IV": stage_iv,
        "Tumor_Aggressiveness_Low": (grade <= 1).astype(int),
        "Tumor_Aggressiveness_Medium": (grade == 2).astype(int),
        "Colonoscopy_Access_Yes": ones,
        "Screening_Regularity_Never": zeros,
        "Screening_Regularity_Regular": screening,
        "Diet_Type_Traditional": zeros,
        "Diet_Type_Western": (1 - activity),
        "Physical_Activity_Level_Low": (1 - activity),
        "Physical_Activity_Level_Medium": zeros,
        "Smoking_Status_Former": zeros,
        "Smoking_Status_Never": (1 - smoker),
        "Alcohol_Consumption_Low": (1 - alcohol),
        "Alcohol_Consumption_Medium": zeros,
        "Red_Meat_Consumption_Low": activity,
        "Red_Meat_Consumption_Medium": (1 - activity),
        "Fiber_Consumption_Low": (1 - activity),
        "Fiber_Consumption_Medium": zeros,
        "Insurance_Coverage_Yes": ones,
        "Time_to_Diagnosis_Timely": ones,
        "Treatment_Access_Limited": zeros,
        "Chemotherapy_Received_Yes": chemo,
        "Radiotherapy_Received_Yes": radiotherapy,
        "Surgery_Received_Yes": surgery,
        "Follow_Up_Adherence_Poor": (1 - follow_up),
    }
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def _reference_to_ui_df(ref_df: pd.DataFrame) -> pd.DataFrame:
    """Converte a coorte de referência para o espaço das 18 variáveis da interface."""
    if all(c in ref_df.columns for c in UI_FEATURE_COLUMNS):
        return ref_df[UI_FEATURE_COLUMNS].copy()

    n = len(ref_df)
    parity = (np.arange(n) % 2).astype(float)
    stage_ii = ref_df["Stage_at_Diagnosis_II"].to_numpy() >= 0.5
    stage_iii = ref_df["Stage_at_Diagnosis_III"].to_numpy() >= 0.5
    stage_iv = ref_df["Stage_at_Diagnosis_IV"].to_numpy() >= 0.5

    t_stage = np.where(
        stage_iv,
        4.0,
        np.where(stage_iii, 3.0, np.where(stage_ii, 3.0 + parity, 1.0 + parity)),
    )
    n_stage = np.where(stage_iv, 2.0, np.where(stage_iii, 1.0 + parity, 0.0))
    m_stage = np.where(stage_iv, 1.0, 0.0)

    inc_low = ref_df["Socioeconomic_Status_Low"].to_numpy() >= 0.5
    inc_mid = ref_df["Socioeconomic_Status_Middle"].to_numpy() >= 0.5
    income = np.where(inc_low, 1.0, np.where(inc_mid, 2.0, 3.0))

    gr_low = ref_df["Tumor_Aggressiveness_Low"].to_numpy() >= 0.5
    gr_mid = ref_df["Tumor_Aggressiveness_Medium"].to_numpy() >= 0.5
    grade = np.where(gr_low, 1.0, np.where(gr_mid, 2.0, 3.0))

    ui_data = {
        "age": ref_df["Age"].to_numpy(dtype=float),
        "sex_male": (ref_df["Gender_Male"].to_numpy(dtype=float) >= 0.5).astype(float),
        "bmi": ref_df["BMI"].to_numpy(dtype=float),
        "income": income,
        "t_stage": t_stage,
        "n_stage": n_stage,
        "m_stage": m_stage,
        "grade": grade,
        "family_history": (ref_df["Family_History_Yes"].to_numpy(dtype=float) >= 0.5).astype(float),
        "previous_cancer": (ref_df["Previous_Cancer_History_Yes"].to_numpy(dtype=float) >= 0.5).astype(float),
        "surgery": (ref_df["Surgery_Received_Yes"].to_numpy(dtype=float) >= 0.5).astype(float),
        "chemo": (ref_df["Chemotherapy_Received_Yes"].to_numpy(dtype=float) >= 0.5).astype(float),
        "radiotherapy": (ref_df["Radiotherapy_Received_Yes"].to_numpy(dtype=float) >= 0.5).astype(float),
        "follow_up": (1.0 - (ref_df["Follow_Up_Adherence_Poor"].to_numpy(dtype=float) >= 0.5).astype(float)),
        "screening": (ref_df["Screening_Regularity_Regular"].to_numpy(dtype=float) >= 0.5).astype(float),
        "smoker": (1.0 - (ref_df["Smoking_Status_Never"].to_numpy(dtype=float) >= 0.5).astype(float)),
        "alcohol": (1.0 - (ref_df["Alcohol_Consumption_Low"].to_numpy(dtype=float) >= 0.5).astype(float)),
        "activity": (1.0 - (ref_df["Physical_Activity_Level_Low"].to_numpy(dtype=float) >= 0.5).astype(float)),
    }
    return pd.DataFrame(ui_data, columns=UI_FEATURE_COLUMNS)


def _log_hazard_ui(model: Any):
    def f(X_ui: np.ndarray) -> np.ndarray:
        df_model = _ui_matrix_to_model_df(X_ui)
        return np.asarray(model.predict_log_partial_hazard(df_model)).ravel()
    return f


def build_explainer(model: Any, background: Optional[pd.DataFrame] = None) -> shap.Explainer:
    """Cria o explicador SHAP exclusivamente com as 18 variáveis que constam na interface."""
    if background is None:
        background = load_reference_cohort(model, n=100)
    ui_bg = _reference_to_ui_df(background)
    masker = shap.maskers.Independent(ui_bg.to_numpy(dtype=float), max_samples=100)
    explainer = shap.Explainer(
        _log_hazard_ui(model),
        masker,
        feature_names=[FRIENDLY_NAMES[c] for c in UI_FEATURE_COLUMNS],
    )
    setattr(explainer, "_ui_schema_version", EXP_SCHEMA_VERSION)
    return explainer


def generate_shap_plot(patient_data: Dict[str, Any], model: Any, explainer: Optional[shap.Explainer] = None):
    """Retorna uma Figura Matplotlib com o gráfico SHAP Waterfall das variáveis da interface."""
    if explainer is None or getattr(explainer, "_ui_schema_version", None) != EXP_SCHEMA_VERSION:
        explainer = build_explainer(model)
    X_ui = encode_ui_patient(patient_data)
    try:
        exp = explainer(X_ui.to_numpy(dtype=float))
    except Exception:
        explainer = build_explainer(model)
        exp = explainer(X_ui.to_numpy(dtype=float))
    plt.close("all")
    fig = plt.figure(figsize=(8.5, 7.0))
    shap.plots.waterfall(exp[0], max_display=18, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.5, 7.0)
    fig.tight_layout()
    return fig


def build_lime_explainer(model: Any, background: Optional[pd.DataFrame] = None):
    """Cria o explicador LIME Tabular exclusivamente com as 18 variáveis da interface."""
    from lime.lime_tabular import LimeTabularExplainer

    if background is None:
        background = load_reference_cohort(model, n=1000)
    ui_bg = _reference_to_ui_df(background)
    cat_idx = [UI_FEATURE_COLUMNS.index(c) for c in BINARY_UI_FEATURES]
    cat_names = {}
    for c in BINARY_UI_FEATURES:
        idx = UI_FEATURE_COLUMNS.index(c)
        if c == "sex_male":
            cat_names[idx] = ["F", "M"]
        elif c == "m_stage":
            cat_names[idx] = ["M0", "M1"]
        else:
            cat_names[idx] = ["Não", "Sim"]

    lime_exp = LimeTabularExplainer(
        ui_bg[UI_FEATURE_COLUMNS].to_numpy(dtype=float),
        feature_names=[FRIENDLY_NAMES[c] for c in UI_FEATURE_COLUMNS],
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
    """Retorna `(Figure, DataFrame)` com a explicação LIME local usando apenas as variáveis da interface."""
    if lime_explainer is None or getattr(lime_explainer, "_ui_schema_version", None) != EXP_SCHEMA_VERSION:
        lime_explainer = build_lime_explainer(model)
    X_ui = encode_ui_patient(patient_data)
    try:
        exp = lime_explainer.explain_instance(
            X_ui.to_numpy(dtype=float)[0],
            _log_hazard_ui(model),
            num_features=num_features,
            num_samples=num_samples,
        )
    except Exception:
        lime_explainer = build_lime_explainer(model)
        exp = lime_explainer.explain_instance(
            X_ui.to_numpy(dtype=float)[0],
            _log_hazard_ui(model),
            num_features=num_features,
            num_samples=num_samples,
        )
    items = exp.as_list()
    df = pd.DataFrame(items, columns=["Condição", "Peso"])
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(df) + 1.4))
    colors = ["#f87171" if w > 0 else "#34d399" for w in df["Peso"]]
    ax.barh(df["Condição"][::-1], df["Peso"][::-1], color=colors[::-1])
    ax.axvline(0, color="#64748b", lw=1)
    ax.set_xlabel("Contribuição para o log-hazard (vermelho = ↑ risco)")
    ax.set_title(f"LIME — fidelidade local R² = {exp.score:.2f}")
    fig.tight_layout()
    return fig, df
