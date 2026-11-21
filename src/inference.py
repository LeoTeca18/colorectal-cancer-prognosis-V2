"""Motor de inferência clínica de sobrevida livre de recidiva e risco (Pilar B).

Totalmente alinhado com o protocolo do Notebook (`Colorectal_Cancer_Prognosis_V2.ipynb`):
- Consome ambos os artefactos exportados: `models/cox_survival_model.pkl` (CoxPH Isolado)
  e `models/clinical_pipeline_bundle.pkl` (LightGBM Otimizado + Platt OOF + Limiar Clínico).
- Mapeia de forma independente as 26 covariáveis brutas para as 43 colunas One-Hot (`drop='first'`).
- Estratifica o risco pela mediana (Baixo vs. Alto Risco) e usa o horizonte máximo observado (59m).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .data_validation import WINSOR_BOUNDS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cox_survival_model.pkl"
BUNDLE_PATH = ROOT / "models" / "clinical_pipeline_bundle.pkl"
REFERENCE_PATH = ROOT / "models" / "reference_cohort.csv"

# Duração máxima observada no treino do CoxPH = 59 meses (1..59)
HORIZONS_MONTHS = (12, 36, 59)
MAX_MONTHS = 59

# Ordem exata das 26 covariáveis brutas do Notebook (2 numéricas + 24 categóricas)
RAW_FEATURE_COLUMNS: List[str] = [
    "age",
    "bmi",
    "gender",
    "race",
    "region",
    "urban_or_rural",
    "socioeconomic_status",
    "family_history",
    "previous_cancer_history",
    "stage_at_diagnosis",
    "tumor_aggressiveness",
    "colonoscopy_access",
    "screening_regularity",
    "diet_type",
    "physical_activity_level",
    "smoking_status",
    "alcohol_consumption",
    "red_meat_consumption",
    "fiber_consumption",
    "insurance_coverage",
    "time_to_diagnosis",
    "treatment_access",
    "chemotherapy_received",
    "radiotherapy_received",
    "surgery_received",
    "follow_up_adherence",
]

# Especificação One-Hot Encoding (drop='first' alfabético, idêntico ao Notebook Fase 2):
# (campo_ui, prefixo_csv, categoria_referencia_omitida, categorias_codificadas)
OHE_SPEC: List[Tuple[str, str, str, Tuple[str, ...]]] = [
    ("gender", "Gender", "Female", ("Male",)),
    ("race", "Race", "Asian", ("Black", "Hispanic", "Other", "White")),
    ("region", "Region", "Africa", ("Asia Pacific", "Europe", "Latin America", "North America")),
    ("urban_or_rural", "Urban_or_Rural", "Rural", ("Urban",)),
    ("socioeconomic_status", "Socioeconomic_Status", "High", ("Low", "Middle")),
    ("family_history", "Family_History", "No", ("Yes",)),
    ("previous_cancer_history", "Previous_Cancer_History", "No", ("Yes",)),
    ("stage_at_diagnosis", "Stage_at_Diagnosis", "I", ("II", "III", "IV")),
    ("tumor_aggressiveness", "Tumor_Aggressiveness", "High", ("Low", "Medium")),
    ("colonoscopy_access", "Colonoscopy_Access", "No", ("Yes",)),
    ("screening_regularity", "Screening_Regularity", "Irregular", ("Never", "Regular")),
    ("diet_type", "Diet_Type", "Balanced", ("Traditional", "Western")),
    ("physical_activity_level", "Physical_Activity_Level", "High", ("Low", "Medium")),
    ("smoking_status", "Smoking_Status", "Current", ("Former", "Never")),
    ("alcohol_consumption", "Alcohol_Consumption", "High", ("Low", "Medium")),
    ("red_meat_consumption", "Red_Meat_Consumption", "High", ("Low", "Medium")),
    ("fiber_consumption", "Fiber_Consumption", "High", ("Low", "Medium")),
    ("insurance_coverage", "Insurance_Coverage", "No", ("Yes",)),
    ("time_to_diagnosis", "Time_to_Diagnosis", "Delayed", ("Timely",)),
    ("treatment_access", "Treatment_Access", "Good", ("Limited",)),
    ("chemotherapy_received", "Chemotherapy_Received", "No", ("Yes",)),
    ("radiotherapy_received", "Radiotherapy_Received", "No", ("Yes",)),
    ("surgery_received", "Surgery_Received", "No", ("Yes",)),
    ("follow_up_adherence", "Follow_Up_Adherence", "Good", ("Poor",)),
]

# Exatas 43 colunas internas esperadas pelos modelos treinados no Notebook
FEATURE_COLUMNS: List[str] = ["Age", "BMI"] + [
    f"{csv_prefix}_{cat}"
    for _, csv_prefix, _, cats in OHE_SPEC
    for cat in cats
]

# Exatos 8 fatores modificáveis da Simulação Contrafactual What-If do Notebook (Fase 5.3)
INTERVENTIONS: Dict[str, tuple] = {
    "follow_up": ("Adesão rigorosa ao acompanhamento clínico", "follow_up_adherence", "Good"),
    "screening": ("Rastreio regular por colonoscopia", "screening_regularity", "Regular"),
    "activity": ("Atividade física elevada", "physical_activity_level", "High"),
    "smoking": ("Cessação tabágica completa", "smoking_status", "Never"),
    "alcohol": ("Redução do consumo de álcool", "alcohol_consumption", "Low"),
    "red_meat": ("Redução do consumo de carne vermelha", "red_meat_consumption", "Low"),
    "fiber": ("Dieta rica em fibras", "fiber_consumption", "High"),
    "bmi": ("Normalização ponderal (IMC ≤ 22.5 kg/m²)", "bmi", 22.5),
}


def load_model(path: Path = MODEL_PATH) -> Optional[Any]:
    """Carrega o modelo de sobrevivência CoxPH exportado do Notebook."""
    if not Path(path).exists():
        return None
    return joblib.load(path)


def load_bundle(path: Path = BUNDLE_PATH) -> Optional[Dict[str, Any]]:
    """Carrega o classificador campeão (`LightGBM + Platt OOF`) e métricas validadas."""
    if not Path(path).exists():
        return None
    return joblib.load(path)


load_pipeline_bundle = load_bundle


def to_logit(p: np.ndarray | float) -> np.ndarray:
    """Transformação logit idêntica à calibração de Platt OOF da Fase 3."""
    p_clip = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)


def encode_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Converte um DataFrame com as 26 colunas brutas nas 43 colunas One-Hot do modelo."""
    age_lo, age_hi = WINSOR_BOUNDS["Age"]
    bmi_lo, bmi_hi = WINSOR_BOUNDS["BMI"]

    encoded: Dict[str, np.ndarray] = {
        "Age": np.clip(df_raw["age"].to_numpy(dtype=float), age_lo, age_hi),
        "BMI": np.clip(df_raw["bmi"].to_numpy(dtype=float), bmi_lo, bmi_hi),
    }
    for field, csv_prefix, _, cats in OHE_SPEC:
        col_vals = df_raw[field].astype(str).to_numpy()
        for cat in cats:
            encoded[f"{csv_prefix}_{cat}"] = (col_vals == cat).astype(float)

    return pd.DataFrame(encoded, columns=FEATURE_COLUMNS)


def encode_patient(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """Converte um dicionário clínico validado (26 campos) na linha de 43 features do modelo."""
    return encode_dataframe(pd.DataFrame([patient_data]))


def decode_ohe_to_ui(df_ohe: pd.DataFrame) -> pd.DataFrame:
    """Reconstrói de forma 100% exata as 26 covariáveis brutas a partir das 43 colunas One-Hot."""
    decoded: Dict[str, Any] = {
        "age": df_ohe["Age"].to_numpy(dtype=float),
        "bmi": df_ohe["BMI"].to_numpy(dtype=float),
    }
    n = len(df_ohe)
    for field, csv_prefix, ref_cat, cats in OHE_SPEC:
        vals = np.full(n, ref_cat, dtype=object)
        for cat in cats:
            col = f"{csv_prefix}_{cat}"
            if col in df_ohe.columns:
                mask = df_ohe[col].to_numpy(dtype=float) >= 0.5
                vals = np.where(mask, cat, vals)
        decoded[field] = vals
    return pd.DataFrame(decoded, columns=RAW_FEATURE_COLUMNS)


def predict_recurrence_risk(patient_data: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Prediz probabilidade bruta, calibrada (Platt OOF) e classe de risco via LightGBM campeão."""
    X = encode_patient(patient_data)
    clf = bundle["best_clf"]
    calibrator = bundle["platt_calibrator"]
    thr = float(bundle["final_thr"])

    prob_raw = float(clf.predict_proba(X)[:, 1][0])
    prob_cal = float(calibrator.predict_proba(to_logit(prob_raw))[:, 1][0])
    return {
        "model_name": bundle.get("best_model_name", "LightGBM (Otimizado)"),
        "prob_raw": prob_raw,
        "prob_calibrated": prob_cal,
        "threshold": thr,
        "high_risk": prob_raw >= thr,
    }


def predict_survival_curve(
    patient_data: Dict[str, Any], model: Any, max_months: int = MAX_MONTHS
) -> pd.DataFrame:
    """Retorna `DataFrame[month, survival]` livre de recidiva para os meses `0..max_months`."""
    X = encode_patient(patient_data)
    months = np.arange(0, max_months + 1)
    sf = model.predict_survival_function(X, times=months)
    return pd.DataFrame({"month": months, "survival": sf.iloc[:, 0].to_numpy()})


def survival_at(curve: pd.DataFrame, horizons: Sequence[int] = HORIZONS_MONTHS) -> Dict[int, float]:
    """Probabilidade livre de recidiva nos horizontes solicitados (12, 36, 59 meses)."""
    return {h: float(np.interp(h, curve["month"], curve["survival"])) for h in horizons}


def log_hazard(patient_data: Dict[str, Any], model: Any) -> float:
    """Preditor linear (log partial hazard) de um doente no modelo CoxPH."""
    return float(np.asarray(model.predict_log_partial_hazard(encode_patient(patient_data))).ravel()[0])


def risk_reduction_pct(patient_data: Dict[str, Any], scenario: Dict[str, Any], model: Any) -> float:
    """Redução relativa do hazard parcial (%) do cenário simulado face ao perfil atual (CoxPH)."""
    hr = np.exp(log_hazard(scenario, model) - log_hazard(patient_data, model))
    return float((1.0 - hr) * 100.0)


def counterfactual_summary(
    patient_data: Dict[str, Any],
    scenario: Dict[str, Any],
    model: Any,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Calcula a redução relativa de risco em CoxPH e (se disponível) em LightGBM bruto e Platt OOF."""
    out: Dict[str, float] = {
        "cox_reduction_pct": risk_reduction_pct(patient_data, scenario, model),
    }
    if bundle is not None:
        base_r = predict_recurrence_risk(patient_data, bundle)
        cf_r = predict_recurrence_risk(scenario, bundle)
        out["lgb_raw_factual"] = base_r["prob_raw"]
        out["lgb_raw_cf"] = cf_r["prob_raw"]
        out["lgb_raw_reduction_pct"] = (
            (base_r["prob_raw"] - cf_r["prob_raw"]) / base_r["prob_raw"] * 100.0
            if base_r["prob_raw"] > 0
            else 0.0
        )
        out["lgb_cal_factual"] = base_r["prob_calibrated"]
        out["lgb_cal_cf"] = cf_r["prob_calibrated"]
        out["lgb_cal_reduction_pct"] = (
            (base_r["prob_calibrated"] - cf_r["prob_calibrated"]) / base_r["prob_calibrated"] * 100.0
            if base_r["prob_calibrated"] > 0
            else 0.0
        )
    return out


def is_intervention_applicable(patient_data: Dict[str, Any], key: str) -> bool:
    """Verifica se a intervenção contrafactual modifica o estado atual do doente."""
    _, field, target = INTERVENTIONS[key]
    if key == "bmi":
        return float(patient_data["bmi"]) > float(target)
    return patient_data.get(field) != target


def apply_interventions(patient_data: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Cópia dos dados do doente com as intervenções selecionadas aplicadas (Fase 5.3)."""
    out = dict(patient_data)
    for k in keys:
        _, field, value = INTERVENTIONS[k]
        if k == "bmi":
            out[field] = min(float(out[field]), float(value))
        else:
            out[field] = value
    return out


def load_reference_cohort(model: Any, n: int = 1000) -> pd.DataFrame:
    """Carrega a coorte de referência (`models/reference_cohort.csv`) do Treino."""
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
    """Curvas médias livre de recidiva estratificadas pela mediana (Fase 4: Baixo vs. Alto risco)."""
    cohort = load_reference_cohort(model, n=1000)
    lp = np.asarray(model.predict_log_partial_hazard(cohort)).ravel()
    med = float(np.median(lp))
    groups = {
        "Baixo risco": lp < med,
        "Alto risco": lp >= med,
    }
    months = np.arange(0, max_months + 1)
    curves = {}
    for name, mask in groups.items():
        sf = model.predict_survival_function(cohort[mask], times=months)
        curves[name] = pd.DataFrame({"month": months, "survival": sf.mean(axis=1).to_numpy()})
    p_lp = log_hazard(patient_data, model)
    group = "Alto risco" if p_lp >= med else "Baixo risco"
    return {"curves": curves, "patient_group": group}
