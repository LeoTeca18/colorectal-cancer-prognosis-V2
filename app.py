"""CRC Prognosis & Survival Dashboard (Streamlit entry point).

Arquitetura de Separation of Concerns (Deploy & Interface Clínica — apenas inferência).
Utiliza exclusivamente o modelo vencedor e as melhores métricas extraídas do Notebook.
"""
from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import streamlit as st

import src.data_validation as _dv_mod
import src.inference as _inf_mod
import src.explainability as _exp_mod
import src.ui_components as _ui_mod

# Garante recarregamento limpo dos submódulos se o servidor Streamlit já estiver em execução
importlib.reload(_dv_mod)
importlib.reload(_inf_mod)
importlib.reload(_exp_mod)
importlib.reload(_ui_mod)

from src.data_validation import LIMITS, PatientClinicalData, parse_and_validate
from src.explainability import (
    build_explainer,
    build_lime_explainer,
    generate_lime_plot,
    generate_shap_plot,
)
from src.inference import (
    HORIZONS_MONTHS,
    INTERVENTIONS,
    MODEL_PATH,
    apply_interventions,
    load_model,
    load_reference_cohort,
    predict_survival_curve,
    risk_reduction_pct,
    risk_strata_curves,
    survival_at,
)
from src.ui_components import (
    inject_css,
    render_error,
    render_metric_card,
    risk_reduction_figure,
    strata_figure,
    survival_figure,
)

st.set_page_config(page_title="CRC Prognosis — Decisão Clínica", page_icon="🧬", layout="wide")
inject_css()

LEVELS = {1: "Baixo", 2: "Médio", 3: "Alto"}
SCHEMA_VERSION = "v3_ui_18_explainers"


# ============================================================================
# CARREGAMENTO E CACHE DE ARTEFACTOS (MLOps)
# ============================================================================
@st.cache_resource(show_spinner="A carregar modelo clínico do disco…")
def get_model_from_disk(path: str = str(MODEL_PATH), version: str = SCHEMA_VERSION) -> Optional[Any]:
    return load_model(Path(path))


@st.cache_resource(show_spinner="A deserializar modelo enviado…")
def get_model_from_bytes(uploaded: bytes, version: str = SCHEMA_VERSION) -> Any:
    return joblib.load(io.BytesIO(uploaded))


def get_model(uploaded: bytes | None = None) -> Optional[Any]:
    """Fachada unificada para obtenção do modelo vencedor (disco ou upload)."""
    if uploaded is not None:
        return get_model_from_bytes(uploaded, SCHEMA_VERSION)
    return get_model_from_disk(str(MODEL_PATH), SCHEMA_VERSION)


@st.cache_resource(show_spinner="A construir explicador SHAP…")
def get_shap_explainer_v3(_model: Any, model_signature: int, version: str = SCHEMA_VERSION):
    return build_explainer(_model, load_reference_cohort(_model, n=100))


@st.cache_resource(show_spinner="A construir explicador LIME…")
def get_lime_explainer_v3(_model: Any, model_signature: int, version: str = SCHEMA_VERSION):
    return build_lime_explainer(_model, load_reference_cohort(_model, n=1000))


@st.cache_data(show_spinner="A calcular estratos de risco populacionais…")
def get_strata(_model: Any, patient_key: tuple, patient: Dict[str, Any], version: str = SCHEMA_VERSION):
    return risk_strata_curves(patient, _model)


# ============================================================================
# FORMULÁRIO DE INGESTÃO COM SINGLE-BATCH VALIDATION (Pydantic v2 + st.form)
# ============================================================================
def sidebar_form() -> tuple[Optional[PatientClinicalData], List[str], List[str], Dict[str, Any]]:
    """Renderiza os inputs clínicos dentro de um st.form para Single-Batch Validation."""
    sb = st.sidebar
    sb.header("🩺 Dados do Doente")

    with sb.form("patient_clinical_form", clear_on_submit=False):
        with st.expander("👤 Demográfica", expanded=True):
            age = st.number_input(
                "Idade (anos)",
                min_value=int(LIMITS["age"][0]),
                max_value=int(LIMITS["age"][1]),
                value=62,
                step=1,
                help="Faixa etária do modelo: 18 a 90 anos.",
            )
            sex = st.radio("Sexo Biológico", ["F", "M"], horizontal=True)
            bmi = st.number_input(
                "IMC (kg/m²)",
                min_value=float(LIMITS["bmi"][0]),
                max_value=float(LIMITS["bmi"][1]),
                value=26.0,
                step=0.1,
                help="Faixa biológica aceitável: 14.0 a 60.0 kg/m².",
            )

        with st.expander("💶 Socioeconómica", expanded=False):
            income = st.select_slider("Nível socioeconómico", [1, 2, 3], value=2, format_func=LEVELS.get)

        with st.expander("🔬 Clínica e Patologia", expanded=True):
            c1, c2, c3 = st.columns(3)
            t = c1.selectbox("Estádio T", [1, 2, 3, 4], index=2)
            n = c2.selectbox("Estádio N", [0, 1, 2], index=1)
            m = c3.selectbox("Estádio M", [0, 1], index=0)
            grade = st.selectbox("Grau / agressividade tumoral", [1, 2, 3], index=1, format_func=LEVELS.get)
            family_history = int(st.checkbox("História familiar de CCR", value=False))
            previous_cancer = int(st.checkbox("História prévia de cancro", value=False))

        with st.expander("💊 Tratamento", expanded=False):
            surgery = int(st.checkbox("Cirurgia realizada", value=True))
            chemo = int(st.checkbox("Adesão à quimioterapia", value=False))
            radiotherapy = int(st.checkbox("Radioterapia", value=False))

        with st.expander("🚬 Acompanhamento e Estilo de Vida", expanded=False):
            follow_up = int(st.checkbox("Adesão rigorosa ao follow-up", value=False))
            screening = int(st.checkbox("Rastreio regular por colonoscopia", value=False))
            smoker = int(st.checkbox("Fumador ativo", value=True))
            alcohol = int(st.checkbox("Consumo elevado de álcool", value=False))
            activity = int(st.checkbox("Atividade física regular", value=False))

        submit_btn = st.form_submit_button("🩺 Avaliar Doente", width="stretch")

    raw_payload: Dict[str, Any] = dict(
        age=age,
        sex=sex,
        bmi=bmi,
        income=income,
        t_stage=t,
        n_stage=n,
        m_stage=m,
        grade=grade,
        family_history=family_history,
        previous_cancer=previous_cancer,
        surgery=surgery,
        chemo=chemo,
        radiotherapy=radiotherapy,
        follow_up=follow_up,
        screening=screening,
        smoker=smoker,
        alcohol=alcohol,
        activity=activity,
    )

    cached_patient = st.session_state.get("validated_patient")
    schema_changed = cached_patient is None or not hasattr(cached_patient, "sex")

    if submit_btn or schema_changed or st.session_state.get("schema_version") != SCHEMA_VERSION:
        validated_obj, errs, warns = parse_and_validate(raw_payload)
        st.session_state["validated_patient"] = validated_obj
        st.session_state["patient_errors"] = errs
        st.session_state["patient_warnings"] = warns
        st.session_state["raw_payload"] = raw_payload
        st.session_state["schema_version"] = SCHEMA_VERSION

    return (
        st.session_state.get("validated_patient"),
        st.session_state.get("patient_errors", []),
        st.session_state.get("patient_warnings", []),
        st.session_state.get("raw_payload", raw_payload),
    )


def sidebar_simulator(patient: Dict[str, Any]) -> List[str]:
    """Intervenções modificáveis selecionadas pelo clínico (cenários What-If)."""
    st.sidebar.divider()
    st.sidebar.header("🔮 Simulação de Cenários")
    applicable = [k for k, (_, field, value) in INTERVENTIONS.items() if patient.get(field) != value]
    if not applicable:
        st.sidebar.info("Todas as intervenções modificáveis já foram adotadas.")
        return []
    # Remove escolhas antigas da sessão que não pertençam às chaves atuais
    if "sim_choice" in st.session_state:
        st.session_state["sim_choice"] = [
            k for k in st.session_state["sim_choice"] if k in applicable
        ]
    return st.sidebar.multiselect(
        "Intervenções a simular",
        applicable,
        format_func=lambda k: INTERVENTIONS[k][0],
        key="sim_choice",
    )


# ============================================================================
# CONTROLADOR PRINCIPAL
# ============================================================================
def main() -> None:
    st.title("🧬 Prognóstico e Sobrevivência — Cancro Colorretal")

    model = get_model()

    if model is None:
        st.error(
            f"Modelo não encontrado em `{MODEL_PATH}`.\n\n"
            "**Porquê:** a aplicação opera em 'Separation of Concerns' (apenas inferência); "
            "necessita do artefacto `cox_survival_model.pkl` previamente treinado.\n\n"
            "**Solução:** carregue o ficheiro abaixo ou coloque-o na pasta `models/`."
        )
        up = st.file_uploader("Carregue o ficheiro `cox_survival_model.pkl`", type=["pkl"])
        if up is None:
            st.info("A aplicação ficará em espera até o modelo estar disponível.")
            st.stop()
        try:
            model = get_model(up.getvalue())
        except Exception as exc:  # noqa: BLE001
            render_error("Não foi possível ler o ficheiro carregado.", exc)
            st.stop()

    validated_patient, errors, warnings, raw_patient = sidebar_form()
    is_valid = validated_patient is not None and not errors
    patient_dict = validated_patient.model_dump() if validated_patient else raw_patient

    chosen = sidebar_simulator(patient_dict) if is_valid else []

    tab_overview, tab_pred, tab_xai, tab_sim = st.tabs(
        ["Visão Geral", "Previsão", "Explicabilidade", "Simulação"]
    )

    with tab_overview:
        st.subheader("Resumo do doente")
        for msg in errors:
            st.error(msg)
        for msg in warnings:
            st.warning(msg)
        if is_valid:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Idade", f"{validated_patient.age} anos")
            c2.metric("TNM", f"T{validated_patient.t_stage}N{validated_patient.n_stage}M{validated_patient.m_stage}")
            c3.metric("Estádio AJCC", validated_patient.stage_ajcc)
            c4.metric("IMC", f"{validated_patient.bmi:.1f} kg/m²")
        else:
            st.error("⛔ Dados com inconformidade clínica. A inferência está bloqueada para prevenir predições enviesadas.")

        st.caption(
            f"Limites aceites: idade [{LIMITS['age'][0]:.0f}-{LIMITS['age'][1]:.0f} anos], "
            f"IMC [{LIMITS['bmi'][0]:.1f}-{LIMITS['bmi'][1]:.1f} kg/m²]."
        )
        st.info("Ferramenta de apoio à decisão clínica baseada em IA; não substitui o julgamento médico.")

    if not is_valid:
        for tab in (tab_pred, tab_xai, tab_sim):
            with tab:
                st.error(
                    "**Esta secção está bloqueada porque os dados do doente não passaram na "
                    "validação Pydantic.** Executar inferências com valores fora dos limites geraria "
                    f"projeções erróneas ({len(errors)} problema(s) detetado(s)):"
                )
                for msg in errors:
                    st.markdown(f"- {msg}")
                st.caption("Ajuste os campos na barra lateral e prima 'Avaliar Doente' para recalcular.")
        return

    try:
        curve = predict_survival_curve(patient_dict, model)
        surv = survival_at(curve)
    except Exception as exc:  # noqa: BLE001
        with tab_pred:
            render_error("Falha ao calcular a curva de sobrevivência.", exc)
        return

    with tab_pred:
        cols = st.columns(len(HORIZONS_MONTHS))
        for col, h in zip(cols, HORIZONS_MONTHS):
            with col:
                render_metric_card(f"Sobrevivência a {h // 12} ano(s)", surv[h])
        st.subheader("Curvas de sobrevivência estratificadas por risco")
        try:
            strata = get_strata(model, tuple(sorted(patient_dict.items())), patient_dict)
            st.info(f"O doente pertence ao estrato de: **{strata['patient_group']}**.")
            st.plotly_chart(strata_figure(strata, curve), width="stretch")
        except Exception as exc:  # noqa: BLE001
            render_error(
                "Estratificação por risco indisponível; a mostrar apenas a curva individual.",
                exc,
                warning=True,
            )
            st.plotly_chart(survival_figure(curve), width="stretch")

    with tab_xai:
        st.subheader("Fatores que influenciam o risco (Explicabilidade Local)")
        st.caption("Valores positivos aumentam o risco (log-hazard); valores negativos reduzem o risco.")
        sub_shap, sub_lime = st.tabs(["SHAP (waterfall)", "LIME (surrogate local)"])
        with sub_shap:
            try:
                explainer = get_shap_explainer_v3(model, id(model), SCHEMA_VERSION)
                with st.spinner("A calcular valores SHAP locais…"):
                    fig = generate_shap_plot(patient_dict, model, explainer)
                st.pyplot(fig)
            except Exception as exc:  # noqa: BLE001
                render_error("Não foi possível gerar a explicação SHAP.", exc)
        with sub_lime:
            st.caption("O LIME ajusta um modelo linear interpretável em torno deste doente específico.")
            try:
                lime_exp = get_lime_explainer_v3(model, id(model), SCHEMA_VERSION)
                with st.spinner("A calcular explicação LIME local…"):
                    lfig, ldf = generate_lime_plot(patient_dict, model, lime_exp)
                st.pyplot(lfig)
                with st.expander("Tabela de pesos locais LIME"):
                    st.dataframe(ldf, width="stretch", hide_index=True)
            except Exception as exc:  # noqa: BLE001
                render_error("Não foi possível gerar a explicação LIME.", exc)

    with tab_sim:
        st.subheader("Simulador contrafactual (What-If)")
        st.caption("Apenas os fatores selecionados são alterados; as restantes variáveis mantêm-se fixas.")
        options = [k for k, (_, f, v) in INTERVENTIONS.items() if patient_dict.get(f) != v]
        if not options:
            st.info("Todas as intervenções modificáveis já estão aplicadas.")
        else:
            labels = [INTERVENTIONS[k][0] for k in options]
            values = [
                risk_reduction_pct(patient_dict, apply_interventions(patient_dict, [k]), model)
                for k in options
            ]
            highlight = None
            if chosen:
                combo_label = "Combinação selecionada"
                labels.append(combo_label)
                values.append(
                    risk_reduction_pct(patient_dict, apply_interventions(patient_dict, chosen), model)
                )
                highlight = combo_label
                scenario = apply_interventions(patient_dict, chosen)
                sim_curve = predict_survival_curve(scenario, model)
                sim = survival_at(sim_curve)
                cols = st.columns(len(HORIZONS_MONTHS))
                for col, h in zip(cols, HORIZONS_MONTHS):
                    with col:
                        render_metric_card(f"{h // 12} ano(s) — simulado", sim[h], sim[h] - surv[h])
                st.plotly_chart(survival_figure(curve, sim_curve), width="stretch")
            else:
                st.info("Escolha intervenções na barra lateral para projetar a curva simulada.")
            st.markdown("#### Redução percentual do risco por intervenção")
            st.plotly_chart(risk_reduction_figure(labels, values, highlight), width="stretch")


if __name__ == "__main__":
    main()
