"""Motor de inferência de sobrevivência (Pilar B).

Recebe os dados clínicos limpos da interface web e converte-os internamente para
o formato esperado pelo modelo vencedor exportado do Notebook (`models/cox_survival_model.pkl`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .data_validation import WINSOR_BOUNDS, tnm_to_stage

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cox_survival_model.pkl"
BUNDLE_PATH = ROOT / "models" / "clinical_pipeline_bundle.pkl"
REFERENCE_PATH = ROOT / "models" / "reference_cohort.csv"

HORIZONS_MONTHS = (12, 36, 60)
MAX_MONTHS = 60

# Colunas internas esperadas pelo modelo treinado no Notebook
FEATURE_COLUMNS: List[str] = [
    "Age",
    "BMI",
    "Gender_Male",
    "Race_Black",
    "Race_Hispanic",
    "Race_Other",
    "Race_White",
    "Region_Asia Pacific",
    "Region_Europe",
    "Region_Latin America",
    "Region_North America",
    "Urban_or_Rural_Urban",
    "Socioeconomic_Status_Low",
    "Socioeconomic_Status_Middle",
    "Family_History_Yes",
    "Previous_Cancer_History_Yes",
    "Stage_at_Diagnosis_II",
    "Stage_at_Diagnosis_III",
    "Stage_at_Diagnosis_IV",
    "Tumor_Aggressiveness_Low",
    "Tumor_Aggressiveness_Medium",
    "Colonoscopy_Access_Yes",
    "Screening_Regularity_Never",
    "Screening_Regularity_Regular",
    "Diet_Type_Traditional",
    "Diet_Type_Western",
    "Physical_Activity_Level_Low",
    "Physical_Activity_Level_Medium",
    "Smoking_Status_Former",
    "Smoking_Status_Never",
    "Alcohol_Consumption_Low",
    "Alcohol_Consumption_Medium",
    "Red_Meat_Consumption_Low",
    "Red_Meat_Consumption_Medium",
    "Fiber_Consumption_Low",
    "Fiber_Consumption_Medium",
    "Insurance_Coverage_Yes",
    "Time_to_Diagnosis_Timely",
    "Treatment_Access_Limited",
    "Chemotherapy_Received_Yes",
    "Radiotherapy_Received_Yes",
    "Surgery_Received_Yes",
    "Follow_Up_Adherence_Poor",
]

# Fatores clínicos modificáveis para a Simulação What-If
INTERVENTIONS: Dict[str, tuple] = {
    "follow_up": ("Adesão rigorosa ao follow-up", "follow_up", 1),
    "screening": ("Rastreio regular por colonoscopia", "screening", 1),
    "smoking": ("Cessação tabágica", "smoker", 0),
    "alcohol": ("Cessação de álcool", "alcohol", 0),
    "activity": ("Atividade física regular", "activity", 1),
    "chemo": ("Adesão à quimioterapia", "chemo", 1),
    "surgery": ("Realização de cirurgia", "surgery", 1),
}


def load_model(path: Path = MODEL_PATH) -> Optional[Any]:
    """Carrega o modelo principal do disco; retorna None se não existir."""
    if not Path(path).exists():
        return None
    return joblib.load(path)


def load_bundle(path: Path = BUNDLE_PATH) -> Optional[Dict[str, Any]]:
    """Carrega os metadados e o classificador campeão exportado do Notebook."""
    if not Path(path).exists():
        return None
    return joblib.load(path)


# Alias de compatibilidade
load_pipeline_bundle = load_bundle


def encode_patient(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """Converte o dicionário clínico limpo da UI na linha de features do modelo."""
    p = patient_data
    age_lo, age_hi = WINSOR_BOUNDS["Age"]
    bmi_lo, bmi_hi = WINSOR_BOUNDS["BMI"]
    age_val = float(np.clip(float(p["age"]), age_lo, age_hi))
    bmi_val = float(np.clip(float(p["bmi"]), bmi_lo, bmi_hi))

    stage = tnm_to_stage(int(p["t_stage"]), int(p["n_stage"]), int(p["m_stage"]))
    grade = int(p["grade"])      # 1=Baixo, 2=Médio, 3=Alto
    income = int(p.get("income", 2))  # 1=Baixo, 2=Médio, 3=Alto

    smoker = int(bool(p["smoker"]))
    alcohol = int(bool(p["alcohol"]))
    activity = int(bool(p["activity"]))
    follow_up = int(bool(p.get("follow_up", 1)))
    screening = int(bool(p.get("screening", 1)))

    row = {
        "Age": age_val,
        "BMI": bmi_val,
        "Gender_Male": int(p["sex"] == "M"),
        # Valores de referência populacional fixos para variáveis administrativas de background
        "Race_Black": 0,
        "Race_Hispanic": 0,
        "Race_Other": 0,
        "Race_White": 1,
        "Region_Asia Pacific": 0,
        "Region_Europe": 1,
        "Region_Latin America": 0,
        "Region_North America": 0,
        "Urban_or_Rural_Urban": 1,
        "Socioeconomic_Status_Low": int(income == 1),
        "Socioeconomic_Status_Middle": int(income == 2),
        "Family_History_Yes": int(bool(p.get("family_history", 0))),
        "Previous_Cancer_History_Yes": int(bool(p.get("previous_cancer", 0))),
        "Stage_at_Diagnosis_II": int(stage == "II"),
        "Stage_at_Diagnosis_III": int(stage == "III"),
        "Stage_at_Diagnosis_IV": int(stage == "IV"),
        "Tumor_Aggressiveness_Low": int(grade == 1),
        "Tumor_Aggressiveness_Medium": int(grade == 2),
        "Colonoscopy_Access_Yes": 1,
        "Screening_Regularity_Never": 0,
        "Screening_Regularity_Regular": int(screening == 1),
        "Diet_Type_Traditional": 0,
        "Diet_Type_Western": int(activity == 0),
        "Physical_Activity_Level_Low": int(activity == 0),
        "Physical_Activity_Level_Medium": 0,
        "Smoking_Status_Former": 0,
        "Smoking_Status_Never": int(smoker == 0),
        "Alcohol_Consumption_Low": int(alcohol == 0),
        "Alcohol_Consumption_Medium": 0,
        "Red_Meat_Consumption_Low": int(activity == 1),
        "Red_Meat_Consumption_Medium": int(activity == 0),
        "Fiber_Consumption_Low": int(activity == 0),
        "Fiber_Consumption_Medium": 0,
        "Insurance_Coverage_Yes": 1,
        "Time_to_Diagnosis_Timely": 1,
        "Treatment_Access_Limited": 0,
        "Chemotherapy_Received_Yes": int(bool(p["chemo"])),
        "Radiotherapy_Received_Yes": int(bool(p["radiotherapy"])),
        "Surgery_Received_Yes": int(bool(p["surgery"])),
        "Follow_Up_Adherence_Poor": int(follow_up == 0),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_survival_curve(
    patient_data: Dict[str, Any], model: Any, max_months: int = MAX_MONTHS
) -> pd.DataFrame:
    """Retorna `DataFrame[month, survival]` para os meses `0..max_months`."""
    X = encode_patient(patient_data)
    months = np.arange(0, max_months + 1)
    sf = model.predict_survival_function(X, times=months)
    return pd.DataFrame({"month": months, "survival": sf.iloc[:, 0].to_numpy()})


def survival_at(curve: pd.DataFrame, horizons: Sequence[int] = HORIZONS_MONTHS) -> Dict[int, float]:
    """Probabilidade de sobrevivência nos horizontes solicitados (12, 36, 60 meses)."""
    return {h: float(np.interp(h, curve["month"], curve["survival"])) for h in horizons}


def log_hazard(patient_data: Dict[str, Any], model: Any) -> float:
    """Preditor linear (log partial hazard) de um doente."""
    return float(np.asarray(model.predict_log_partial_hazard(encode_patient(patient_data))).ravel()[0])


def risk_reduction_pct(patient_data: Dict[str, Any], scenario: Dict[str, Any], model: Any) -> float:
    """Redução relativa de risco (%) do cenário simulado face ao perfil atual."""
    hr = np.exp(log_hazard(scenario, model) - log_hazard(patient_data, model))
    return float((1.0 - hr) * 100.0)


def apply_interventions(patient_data: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Cópia dos dados do doente com as intervenções selecionadas aplicadas."""
    out = dict(patient_data)
    for k in keys:
        _, field, value = INTERVENTIONS[k]
        out[field] = value
    return out


def load_reference_cohort(model: Any, n: int = 600) -> pd.DataFrame:
    """Carrega a coorte de referência (`models/reference_cohort.csv`) para estratificação de risco."""
    if REFERENCE_PATH.exists():
        df = pd.read_csv(REFERENCE_PATH)[FEATURE_COLUMNS]
        return df.sample(min(n, len(df)), random_state=0).reset_index(drop=True)
    rng = np.random.default_rng(0)
    mean = model._norm_mean[FEATURE_COLUMNS]
    std = getattr(model, "_norm_std", pd.Series(1.0, index=FEATURE_COLUMNS))[FEATURE_COLUMNS]
    return pd.DataFrame(
        rng.normal(mean.to_numpy(), std.to_numpy(), (n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )


def risk_strata_curves(
    patient_data: Dict[str, Any], model: Any, max_months: int = MAX_MONTHS
) -> Dict[str, Any]:
    """Curvas médias de sobrevivência por tercis de risco (Baixo, Intermédio, Alto) + grupo do doente."""
    cohort = load_reference_cohort(model)
    lp = np.asarray(model.predict_log_partial_hazard(cohort)).ravel()
    q1, q2 = np.quantile(lp, [1 / 3, 2 / 3])
    groups = {
        "Baixo risco": lp <= q1,
        "Risco intermédio": (lp > q1) & (lp <= q2),
        "Alto risco": lp > q2,
    }
    months = np.arange(0, max_months + 1)
    curves = {}
    for name, mask in groups.items():
        sf = model.predict_survival_function(cohort[mask], times=months)
        curves[name] = pd.DataFrame({"month": months, "survival": sf.mean(axis=1).to_numpy()})
    p_lp = log_hazard(patient_data, model)
    group = "Baixo risco" if p_lp <= q1 else "Risco intermédio" if p_lp <= q2 else "Alto risco"
    return {"curves": curves, "patient_group": group}
