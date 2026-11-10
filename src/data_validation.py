"""Validação clínica de entradas em tempo de inferência (Pilar A) com Pydantic v2.

Focado exclusivamente na validação clínica e plausibilidade médica para a interface web,
mantendo compatibilidade direta com o pipeline treinado no Notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

LIMITS: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 90.0),
    "bmi": (14.0, 60.0),
}

# Limites aprendidos no treino (5%–95%) aplicados silenciosamente pelo motor de inferência
WINSOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Age": (23.0, 86.0),
    "BMI": (19.6, 38.9),
}

VALID_T = (1, 2, 3, 4)
VALID_N = (0, 1, 2)
VALID_M = (0, 1)
VALID_GRADE = (1, 2, 3)
VALID_SEX = ("F", "M")
VALID_LEVEL = (1, 2, 3)
BINARY_FIELDS = (
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
)


def is_in_range(name: str, value: Any) -> bool:
    """Verificação de intervalo numérico contra LIMITS."""
    lo, hi = LIMITS[name.lower()]
    try:
        return lo <= float(value) <= hi
    except (TypeError, ValueError):
        return False


def is_valid_tnm(t: Any, n: Any, m: Any) -> bool:
    """Retorna True se a tríade TNM contém apenas categorias autorizadas."""
    return t in VALID_T and n in VALID_N and m in VALID_M


def tnm_to_stage(t: int, n: int, m: int) -> str:
    """Classificação simplificada AJCC para o estadiamento do cancro colorretal (I a IV)."""
    if m == 1:
        return "IV"
    if n > 0:
        return "III"
    if t >= 3:
        return "II"
    return "I"


class PatientClinicalData(BaseModel):
    """Schema Pydantic v2 para validação clínica do doente na interface web."""

    # Demográficas e Antropométricas
    age: int = Field(..., ge=18, le=90, description="Idade do doente (18 a 90 anos)")
    sex: Literal["F", "M"] = Field(..., description="Sexo biológico (F ou M)")
    bmi: float = Field(..., ge=14.0, le=60.0, description="Índice de Massa Corporal (14.0 a 60.0 kg/m²)")
    income: Literal[1, 2, 3] = Field(default=2, description="Nível socioeconómico (1=Baixo, 2=Médio, 3=Alto)")

    # Estadiamento TNM, Patologia e Antecedentes
    t_stage: Literal[1, 2, 3, 4] = Field(..., description="Estádio T (1-4)")
    n_stage: Literal[0, 1, 2] = Field(..., description="Estádio N (0-2)")
    m_stage: Literal[0, 1] = Field(..., description="Estádio M (0-1)")
    grade: Literal[1, 2, 3] = Field(..., description="Grau / agressividade tumoral (1=Baixo, 2=Médio, 3=Alto)")
    family_history: Literal[0, 1] = Field(default=0, description="História familiar de CCR (0=Não, 1=Sim)")
    previous_cancer: Literal[0, 1] = Field(default=0, description="História prévia de cancro (0=Não, 1=Sim)")

    # Tratamento
    surgery: Literal[0, 1] = Field(..., description="Cirurgia de resseção efetuada (0=Não, 1=Sim)")
    chemo: Literal[0, 1] = Field(..., description="Adesão à quimioterapia (0=Não, 1=Sim)")
    radiotherapy: Literal[0, 1] = Field(..., description="Realização de radioterapia (0=Não, 1=Sim)")

    # Acompanhamento e Estilo de Vida
    follow_up: Literal[0, 1] = Field(default=1, description="Adesão rigorosa ao follow-up (0=Não, 1=Sim)")
    screening: Literal[0, 1] = Field(default=1, description="Rastreio regular por colonoscopia (0=Não, 1=Sim)")
    smoker: Literal[0, 1] = Field(..., description="Fumador ativo (0=Não, 1=Sim)")
    alcohol: Literal[0, 1] = Field(..., description="Consumo elevado de álcool (0=Não, 1=Sim)")
    activity: Literal[0, 1] = Field(..., description="Atividade física regular (0=Não, 1=Sim)")

    @property
    def stage_ajcc(self) -> str:
        """Agrupamento por estádios AJCC (I a IV)."""
        return tnm_to_stage(self.t_stage, self.n_stage, self.m_stage)

    def get_clinical_warnings(self) -> List[str]:
        """Avisos não-bloqueantes de plausibilidade biológica e coerência médica."""
        warnings: List[str] = []
        if self.stage_ajcc == "I" and self.chemo == 1:
            warnings.append("Aviso: Quimioterapia no Estádio I habitualmente não é indicada salvo alto risco específico.")
        if self.m_stage == 0 and self.surgery == 0:
            warnings.append("Alerta: Doença M0 sem cirurgia registada; confirme o plano terapêutico primário.")
        if self.age < 30:
            warnings.append("Alerta Genético: CCR com idade < 30 anos é raro; considere investigação de síndromes hereditárias (ex.: Lynch).")
        if self.bmi < 18.5 or self.bmi > 40.0:
            warnings.append("Aviso Nutricional: IMC em faixa extrema (baixo peso ou obesidade severa); interpretar com cautela clínica.")
        return warnings


@dataclass
class ValidationResult:
    """Resultado da validação de um registo de doente."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_patient: Optional[PatientClinicalData] = None

    @property
    def is_valid(self) -> bool:
        return not self.errors and self.validated_patient is not None


def parse_and_validate(data: Dict[str, Any]) -> Tuple[Optional[PatientClinicalData], List[str], List[str]]:
    """Valida um dicionário através do schema Pydantic v2."""
    field_translations = {
        "age": "Idade",
        "sex": "Sexo",
        "bmi": "IMC",
        "income": "Nível socioeconómico",
        "t_stage": "Estádio T",
        "n_stage": "Estádio N",
        "m_stage": "Estádio M",
        "grade": "Grau tumoral",
        "family_history": "História familiar",
        "previous_cancer": "Cancro prévio",
        "surgery": "Cirurgia",
        "chemo": "Quimioterapia",
        "radiotherapy": "Radioterapia",
        "follow_up": "Adesão ao follow-up",
        "screening": "Rastreio regular",
        "smoker": "Tabagismo",
        "alcohol": "Álcool",
        "activity": "Atividade física",
    }

    try:
        patient = PatientClinicalData(**data)
        warnings = patient.get_clinical_warnings()
        return patient, [], warnings
    except ValidationError as exc:
        formatted_errors: List[str] = []
        for err in exc.errors():
            loc = err["loc"][0] if err["loc"] else "Geral"
            field_name = field_translations.get(str(loc), str(loc))
            msg = err.get("msg", "Valor inválido")
            if "greater_than_equal" in err.get("type", "") or "less_than_equal" in err.get("type", ""):
                lo, hi = LIMITS.get(str(loc), (None, None))
                if lo is not None:
                    formatted_errors.append(
                        f"{field_name}: valor fora do intervalo permitido [{lo:.0f} - {hi:.0f}]. "
                        "O modelo apenas aceita entradas dentro da distribuição clínica de treino."
                    )
                else:
                    formatted_errors.append(f"{field_name}: {msg}.")
            elif "literal_error" in err.get("type", ""):
                formatted_errors.append(
                    f"{field_name}: categoria inválida. Valores aceites: {err.get('ctx', {}).get('expected', 'verifique o campo')}."
                )
            else:
                formatted_errors.append(f"{field_name}: {msg}.")
        return None, formatted_errors, []


def validate_patient(data: Dict[str, Any]) -> ValidationResult:
    """Valida um registo de doente via Pydantic; deve ser invocado antes de qualquer inferência."""
    patient, errors, warnings = parse_and_validate(data)
    return ValidationResult(errors=errors, warnings=warnings, validated_patient=patient)
