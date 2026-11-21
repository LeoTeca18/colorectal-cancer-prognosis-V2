"""Validação clínica de entradas em tempo de inferência (Pilar A) com Pydantic v2.

Alinhado integralmente com as 26 covariáveis independentes do Notebook
(`Colorectal_Cancer_Prognosis_V2.ipynb`) e com os limites aprendidos no treino.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

# Limites de validação na ingestão (alinhados com Fase 1 do Notebook + faixa clínica de IMC)
LIMITS: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 90.0),
    "bmi": (14.0, 60.0),
}

# Limites exatos de winsorização (percentis 5%–95%) aprendidos no Treino (80%) na Fase 2
WINSOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Age": (23.0, 86.0),
    "BMI": (19.6, 38.9),
}

VALID_GENDER = ("Female", "Male")
VALID_RACE = ("Asian", "Black", "Hispanic", "Other", "White")
VALID_REGION = ("Africa", "Asia Pacific", "Europe", "Latin America", "North America")
VALID_URBAN_RURAL = ("Rural", "Urban")
VALID_LEVEL = ("Low", "Medium", "High")
VALID_SOCIOECONOMIC = ("Low", "Middle", "High")
VALID_YN = ("No", "Yes")
VALID_STAGE = ("I", "II", "III", "IV")
VALID_SCREENING = ("Never", "Irregular", "Regular")
VALID_DIET = ("Balanced", "Traditional", "Western")
VALID_SMOKING = ("Never", "Former", "Current")
VALID_TIME_DIAG = ("Timely", "Delayed")
VALID_TX_ACCESS = ("Good", "Limited")
VALID_FOLLOW_UP = ("Good", "Poor")


def is_in_range(name: str, value: Any) -> bool:
    """Verificação de intervalo numérico contra LIMITS."""
    lo, hi = LIMITS[name.lower()]
    try:
        return lo <= float(value) <= hi
    except (TypeError, ValueError):
        return False


class PatientClinicalData(BaseModel):
    """Schema Pydantic v2 para as exatas 26 covariáveis brutas do Notebook."""

    # Demográficas, Antropométricas e Socioeconómicas
    age: int = Field(..., ge=18, le=90, description="Idade do doente (18 a 90 anos)")
    bmi: float = Field(..., ge=14.0, le=60.0, description="Índice de Massa Corporal (14.0 a 60.0 kg/m²)")
    gender: Literal["Female", "Male"] = Field(..., description="Sexo biológico")
    race: Literal["Asian", "Black", "Hispanic", "Other", "White"] = Field(default="White", description="Grupo étnico/racial")
    region: Literal["Africa", "Asia Pacific", "Europe", "Latin America", "North America"] = Field(
        default="Europe", description="Região geográfica"
    )
    urban_or_rural: Literal["Rural", "Urban"] = Field(default="Urban", description="Área de residência")
    socioeconomic_status: Literal["Low", "Middle", "High"] = Field(
        default="Middle", description="Nível socioeconómico"
    )
    insurance_coverage: Literal["No", "Yes"] = Field(default="Yes", description="Cobertura de seguro de saúde")

    # Antecedentes, Estadiamento e Patologia
    family_history: Literal["No", "Yes"] = Field(default="No", description="História familiar de CCR")
    previous_cancer_history: Literal["No", "Yes"] = Field(default="No", description="História prévia de cancro")
    stage_at_diagnosis: Literal["I", "II", "III", "IV"] = Field(..., description="Estádio ao diagnóstico (I–IV)")
    tumor_aggressiveness: Literal["Low", "Medium", "High"] = Field(
        default="Medium", description="Agressividade tumoral"
    )

    # Acesso, Diagnóstico e Terapêutica
    colonoscopy_access: Literal["No", "Yes"] = Field(default="Yes", description="Acesso a colonoscopia")
    time_to_diagnosis: Literal["Timely", "Delayed"] = Field(default="Timely", description="Tempo até ao diagnóstico")
    treatment_access: Literal["Good", "Limited"] = Field(default="Good", description="Acesso ao tratamento")
    surgery_received: Literal["No", "Yes"] = Field(..., description="Cirurgia realizada")
    chemotherapy_received: Literal["No", "Yes"] = Field(..., description="Quimioterapia realizada")
    radiotherapy_received: Literal["No", "Yes"] = Field(..., description="Radioterapia realizada")

    # Rastreio, Seguimento e Estilo de Vida
    follow_up_adherence: Literal["Good", "Poor"] = Field(default="Good", description="Adesão ao follow-up")
    screening_regularity: Literal["Never", "Irregular", "Regular"] = Field(
        default="Regular", description="Regularidade do rastreio"
    )
    diet_type: Literal["Balanced", "Traditional", "Western"] = Field(
        default="Western", description="Padrão alimentar"
    )
    physical_activity_level: Literal["Low", "Medium", "High"] = Field(
        default="Low", description="Nível de atividade física"
    )
    smoking_status: Literal["Never", "Former", "Current"] = Field(
        default="Never", description="Estado tabágico"
    )
    alcohol_consumption: Literal["Low", "Medium", "High"] = Field(
        default="Low", description="Consumo de álcool"
    )
    red_meat_consumption: Literal["Low", "Medium", "High"] = Field(
        default="Medium", description="Consumo de carne vermelha"
    )
    fiber_consumption: Literal["Low", "Medium", "High"] = Field(
        default="Low", description="Consumo de fibra alimentar"
    )

    def get_clinical_warnings(self) -> List[str]:
        """Avisos não-bloqueantes de plausibilidade biológica e limites de winsorização."""
        warnings: List[str] = []
        if self.stage_at_diagnosis == "I" and self.chemotherapy_received == "Yes":
            warnings.append(
                "Aviso: Quimioterapia no Estádio I habitualmente não é indicada salvo alto risco específico."
            )
        if self.stage_at_diagnosis in ("I", "II", "III") and self.surgery_received == "No":
            warnings.append(
                "Alerta: Doença não metastática (Estádio I–III) sem cirurgia registada; confirme o plano terapêutico."
            )
        if self.age < 30:
            warnings.append(
                "Alerta Genético: CCR com idade < 30 anos é raro; considere investigação de síndromes hereditárias (ex.: Lynch)."
            )
        if self.bmi < 18.5 or self.bmi > 40.0:
            warnings.append(
                "Aviso Nutricional: IMC fora da faixa observada na coorte de treino [18.5–40.0 kg/m²]; interpretar com cautela."
            )
        age_lo, age_hi = WINSOR_BOUNDS["Age"]
        bmi_lo, bmi_hi = WINSOR_BOUNDS["BMI"]
        if not (age_lo <= self.age <= age_hi) or not (bmi_lo <= self.bmi <= bmi_hi):
            warnings.append(
                f"Nota de Pré-processamento: Valores fora dos percentis 5%–95% do treino "
                f"(Idade [{age_lo:.0f}–{age_hi:.0f}], IMC [{bmi_lo:.1f}–{bmi_hi:.1f}]) serão winsorizados antes da inferência."
            )
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


FIELD_TRANSLATIONS: Dict[str, str] = {
    "age": "Idade",
    "bmi": "IMC",
    "gender": "Sexo Biológico",
    "race": "Grupo Étnico/Racial",
    "region": "Região",
    "urban_or_rural": "Zona de Residência",
    "socioeconomic_status": "Nível Socioeconómico",
    "insurance_coverage": "Seguro de Saúde",
    "family_history": "História Familiar de CCR",
    "previous_cancer_history": "História Prévia de Cancro",
    "stage_at_diagnosis": "Estádio ao Diagnóstico",
    "tumor_aggressiveness": "Agressividade Tumoral",
    "colonoscopy_access": "Acesso a Colonoscopia",
    "time_to_diagnosis": "Tempo até Diagnóstico",
    "treatment_access": "Acesso ao Tratamento",
    "surgery_received": "Cirurgia",
    "chemotherapy_received": "Quimioterapia",
    "radiotherapy_received": "Radioterapia",
    "follow_up_adherence": "Adesão ao Acompanhamento",
    "screening_regularity": "Regularidade de Rastreio",
    "diet_type": "Padrão Alimentar",
    "physical_activity_level": "Atividade Física",
    "smoking_status": "Tabagismo",
    "alcohol_consumption": "Consumo de Álcool",
    "red_meat_consumption": "Consumo de Carne Vermelha",
    "fiber_consumption": "Consumo de Fibra",
}

# Tradução das categorias internas para exibição em Português na interface e explicabilidade
CATEGORY_LABELS_PT: Dict[str, Dict[str, str]] = {
    "gender": {"Female": "Feminino", "Male": "Masculino"},
    "race": {
        "Asian": "Asiática",
        "Black": "Negra",
        "Hispanic": "Hispânica",
        "Other": "Outra",
        "White": "Branca",
    },
    "region": {
        "Africa": "África",
        "Asia Pacific": "Ásia-Pacífico",
        "Europe": "Europa",
        "Latin America": "América Latina",
        "North America": "América do Norte",
    },
    "urban_or_rural": {"Rural": "Rural", "Urban": "Urbana"},
    "socioeconomic_status": {"Low": "Baixo", "Middle": "Médio", "High": "Alto"},
    "insurance_coverage": {"No": "Não", "Yes": "Sim"},
    "family_history": {"No": "Não", "Yes": "Sim"},
    "previous_cancer_history": {"No": "Não", "Yes": "Sim"},
    "stage_at_diagnosis": {"I": "I", "II": "II", "III": "III", "IV": "IV"},
    "tumor_aggressiveness": {"Low": "Baixa", "Medium": "Média", "High": "Alta"},
    "colonoscopy_access": {"No": "Não", "Yes": "Sim"},
    "time_to_diagnosis": {"Timely": "Oportuno", "Delayed": "Atrasado"},
    "treatment_access": {"Good": "Bom", "Limited": "Limitado"},
    "surgery_received": {"No": "Não", "Yes": "Sim"},
    "chemotherapy_received": {"No": "Não", "Yes": "Sim"},
    "radiotherapy_received": {"No": "Não", "Yes": "Sim"},
    "follow_up_adherence": {"Good": "Boa", "Poor": "Fraca"},
    "screening_regularity": {"Never": "Nunca", "Irregular": "Irregular", "Regular": "Regular"},
    "diet_type": {"Balanced": "Equilibrada", "Traditional": "Tradicional", "Western": "Ocidental"},
    "physical_activity_level": {"Low": "Baixa", "Medium": "Média", "High": "Alta"},
    "smoking_status": {"Never": "Nunca fumou", "Former": "Ex-fumador", "Current": "Fumador atual"},
    "alcohol_consumption": {"Low": "Baixo", "Medium": "Médio", "High": "Alto"},
    "red_meat_consumption": {"Low": "Baixo", "Medium": "Médio", "High": "Alto"},
    "fiber_consumption": {"Low": "Baixo", "Medium": "Médio", "High": "Alto"},
}


def format_category_pt(field: str, value: str) -> str:
    """Devolve o rótulo em Português de uma categoria clínica."""
    return CATEGORY_LABELS_PT.get(field, {}).get(value, value)


def parse_and_validate(data: Dict[str, Any]) -> Tuple[Optional[PatientClinicalData], List[str], List[str]]:
    """Valida um dicionário através do schema Pydantic v2."""
    try:
        patient = PatientClinicalData(**data)
        warnings = patient.get_clinical_warnings()
        return patient, [], warnings
    except ValidationError as exc:
        formatted_errors: List[str] = []
        for err in exc.errors():
            loc = err["loc"][0] if err["loc"] else "Geral"
            field_name = FIELD_TRANSLATIONS.get(str(loc), str(loc))
            msg = err.get("msg", "Valor inválido")
            err_type = err.get("type", "")
            if "greater_than_equal" in err_type or "less_than_equal" in err_type:
                lo, hi = LIMITS.get(str(loc), (None, None))
                if lo is not None:
                    formatted_errors.append(
                        f"{field_name}: valor fora do intervalo permitido [{lo:.0f} - {hi:.0f}]. "
                        "O modelo apenas aceita entradas dentro da distribuição clínica de treino."
                    )
                else:
                    formatted_errors.append(f"{field_name}: {msg}.")
            elif "literal_error" in err_type:
                formatted_errors.append(
                    f"{field_name}: categoria inválida. Valores aceites: "
                    f"{err.get('ctx', {}).get('expected', 'verifique o campo')}."
                )
            else:
                formatted_errors.append(f"{field_name}: {msg}.")
        return None, formatted_errors, []


def validate_patient(data: Dict[str, Any]) -> ValidationResult:
    """Valida um registo de doente via Pydantic; deve ser invocado antes de qualquer inferência."""
    patient, errors, warnings = parse_and_validate(data)
    return ValidationResult(errors=errors, warnings=warnings, validated_patient=patient)
