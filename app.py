"""CRC Prognosis & Recurrence-Free Survival Dashboard (Streamlit entry point).

Arquitetura de Separation of Concerns (Deploy & Interface Clínica — apenas inferência).
Consome os artefactos vencedores exportados do Notebook (`cox_survival_model.pkl` e
`clinical_pipeline_bundle.pkl`) mantendo 100% de conformidade com as 26 covariáveis
independentes, winsorização 5%–95%, calibração Platt OOF e protocolo temporal do Notebook.
"""
from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import streamlit as st

import src.data_validation as _dv_mod
import src.explainability as _exp_mod
import src.inference as _inf_mod
import src.ui_components as _ui_mod

importlib.reload(_dv_mod)
importlib.reload(_inf_mod)
importlib.reload(_exp_mod)
importlib.reload(_ui_mod)

from src.data_validation import (
    LIMITS,
    VALID_DIET,
    VALID_FOLLOW_UP,
    VALID_GENDER,
    VALID_LEVEL,
    VALID_RACE,
    VALID_REGION,
    VALID_SCREENING,
    VALID_SMOKING,
    VALID_SOCIOECONOMIC,
    VALID_STAGE,
    VALID_TIME_DIAG,
    VALID_TX_ACCESS,
    VALID_URBAN_RURAL,
    VALID_YN,
    WINSOR_BOUNDS,
    PatientClinicalData,
    format_category_pt,
    parse_and_validate,
)
from src.explainability import (
    build_explainer,
    build_lgb_tree_explainer,
    build_lime_explainer,
    generate_lgb_shap_plot,
    generate_lime_plot,
    generate_shap_plot,
)
from src.inference import (
    BUNDLE_PATH,
    HORIZONS_MONTHS,
    INTERVENTIONS,
    MODEL_PATH,
    apply_interventions,
    counterfactual_summary,
    is_intervention_applicable,
    load_bundle,
    load_model,
    load_reference_cohort,
    predict_recurrence_risk,
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

st.set_page_config(
    page_title="CRC Prognosis — Sobrevida Livre de Recidiva",
    page_icon="🧬",
    layout="wide",
)
inject_css()

SCHEMA_VERSION = "v5_raw26_pt_ui"


# ============================================================================
# CARREGAMENTO E CACHE DE ARTEFACTOS (MLOps)
# ============================================================================
@st.cache_resource(show_spinner="A carregar modelo de sobrevida CoxPH…")
def get_model_from_disk(path: str = str(MODEL_PATH), version: str = SCHEMA_VERSION) -> Optional[Any]:
    return load_model(Path(path))


@st.cache_resource(show_spinner="A deserializar modelo enviado…")
def get_model_from_bytes(uploaded: bytes, version: str = SCHEMA_VERSION) -> Any:
    return joblib.load(io.BytesIO(uploaded))


def get_model(uploaded: bytes | None = None) -> Optional[Any]:
    """Fachada unificada para obtenção do modelo CoxPH vencedor (disco ou upload)."""
    if uploaded is not None:
        return get_model_from_bytes(uploaded, SCHEMA_VERSION)
    return get_model_from_disk(str(MODEL_PATH), SCHEMA_VERSION)


@st.cache_resource(show_spinner="A carregar classificador campeão (LightGBM + Platt OOF)…")
def get_bundle_from_disk(path: str = str(BUNDLE_PATH), version: str = SCHEMA_VERSION) -> Optional[Dict[str, Any]]:
    return load_bundle(Path(path))


@st.cache_resource(show_spinner="A construir TreeExplainer SHAP (LightGBM)…")
def get_lgb_explainer(_bundle: Dict[str, Any], bundle_sig: int, version: str = SCHEMA_VERSION):
    return build_lgb_tree_explainer(_bundle)


@st.cache_resource(show_spinner="A construir explicador SHAP (CoxPH)…")
def get_shap_explainer(_model: Any, model_signature: int, version: str = SCHEMA_VERSION):
    return build_explainer(_model, load_reference_cohort(_model, n=100))


@st.cache_resource(show_spinner="A construir explicador LIME (CoxPH)…")
def get_lime_explainer(_model: Any, model_signature: int, version: str = SCHEMA_VERSION):
    return build_lime_explainer(_model, load_reference_cohort(_model, n=1000))


@st.cache_data(show_spinner="A calcular estratos de risco pela mediana…")
def get_strata(_model: Any, patient_key: tuple, patient: Dict[str, Any], version: str = SCHEMA_VERSION):
    return risk_strata_curves(patient, _model)


# ============================================================================
# FORMULÁRIO DE INGESTÃO COM SINGLE-BATCH VALIDATION (26 Variáveis Reais)
# ============================================================================
def sidebar_form() -> tuple[Optional[PatientClinicalData], List[str], List[str], Dict[str, Any]]:
    """Renderiza as 26 covariáveis independentes do Notebook dentro de um `st.form`."""
    sb = st.sidebar
    sb.header("🩺 Dados do Doente (26 Variáveis)")

    with sb.form("patient_clinical_form", clear_on_submit=False):
        with st.expander("👤 Demográfica e Socioeconómica", expanded=True):
            age = st.number_input(
                "Idade (anos)",
                min_value=int(LIMITS["age"][0]),
                max_value=int(LIMITS["age"][1]),
                value=62,
                step=1,
                help="Faixa aceite: 18–90 anos (winsorização treino: 23–86).",
            )
            bmi = st.number_input(
                "IMC (kg/m²)",
                min_value=float(LIMITS["bmi"][0]),
                max_value=float(LIMITS["bmi"][1]),
                value=26.0,
                step=0.1,
                help="Faixa aceite: 14.0–60.0 kg/m² (winsorização treino: 19.6–38.9).",
            )
            gender = st.radio(
                "Sexo Biológico",
                list(VALID_GENDER),
                index=1,
                horizontal=True,
                format_func=lambda v: format_category_pt("gender", v),
            )
            race = st.selectbox(
                "Grupo Étnico/Racial",
                list(VALID_RACE),
                index=4,
                format_func=lambda v: format_category_pt("race", v),
            )
            region = st.selectbox(
                "Região",
                list(VALID_REGION),
                index=2,
                format_func=lambda v: format_category_pt("region", v),
            )
            urban_or_rural = st.radio(
                "Zona de Residência",
                list(VALID_URBAN_RURAL),
                index=1,
                horizontal=True,
                format_func=lambda v: format_category_pt("urban_or_rural", v),
            )
            socioeconomic_status = st.select_slider(
                "Nível Socioeconómico",
                list(VALID_SOCIOECONOMIC),
                value="Middle",
                format_func=lambda v: format_category_pt("socioeconomic_status", v),
            )
            insurance_coverage = st.radio(
                "Seguro de Saúde",
                list(VALID_YN),
                index=1,
                horizontal=True,
                format_func=lambda v: format_category_pt("insurance_coverage", v),
            )

        with st.expander("🔬 Clínica e Patologia", expanded=True):
            stage_at_diagnosis = st.selectbox("Estádio ao Diagnóstico (AJCC)", list(VALID_STAGE), index=2)
            tumor_aggressiveness = st.select_slider(
                "Agressividade Tumoral",
                list(VALID_LEVEL),
                value="Medium",
                format_func=lambda v: format_category_pt("tumor_aggressiveness", v),
            )
            family_history = st.radio(
                "História Familiar de CCR",
                list(VALID_YN),
                index=0,
                horizontal=True,
                format_func=lambda v: format_category_pt("family_history", v),
            )
            previous_cancer_history = st.radio(
                "História Prévia de Cancro",
                list(VALID_YN),
                index=0,
                horizontal=True,
                format_func=lambda v: format_category_pt("previous_cancer_history", v),
            )

        with st.expander("💊 Acesso e Tratamento", expanded=False):
            colonoscopy_access = st.radio(
                "Acesso a Colonoscopia",
                list(VALID_YN),
                index=1,
                horizontal=True,
                format_func=lambda v: format_category_pt("colonoscopy_access", v),
            )
            time_to_diagnosis = st.radio(
                "Tempo até ao Diagnóstico",
                list(VALID_TIME_DIAG),
                index=0,
                horizontal=True,
                format_func=lambda v: format_category_pt("time_to_diagnosis", v),
            )
            treatment_access = st.radio(
                "Acesso ao Tratamento",
                list(VALID_TX_ACCESS),
                index=0,
                horizontal=True,
                format_func=lambda v: format_category_pt("treatment_access", v),
            )
            surgery_received = st.radio(
                "Cirurgia Realizada",
                list(VALID_YN),
                index=1,
                horizontal=True,
                format_func=lambda v: format_category_pt("surgery_received", v),
            )
            chemotherapy_received = st.radio(
                "Quimioterapia Realizada",
                list(VALID_YN),
                index=0,
                horizontal=True,
                format_func=lambda v: format_category_pt("chemotherapy_received", v),
            )
            radiotherapy_received = st.radio(
                "Radioterapia Realizada",
                list(VALID_YN),
                index=0,
                horizontal=True,
                format_func=lambda v: format_category_pt("radiotherapy_received", v),
            )

        with st.expander("🚬 Seguimento e Estilo de Vida", expanded=False):
            follow_up_adherence = st.radio(
                "Adesão ao Acompanhamento",
                list(VALID_FOLLOW_UP),
                index=1,
                horizontal=True,
                format_func=lambda v: format_category_pt("follow_up_adherence", v),
            )
            screening_regularity = st.selectbox(
                "Regularidade de Rastreio",
                list(VALID_SCREENING),
                index=1,
                format_func=lambda v: format_category_pt("screening_regularity", v),
            )
            physical_activity_level = st.select_slider(
                "Atividade Física",
                list(VALID_LEVEL),
                value="Low",
                format_func=lambda v: format_category_pt("physical_activity_level", v),
            )
            smoking_status = st.selectbox(
                "Estado Tabágico",
                list(VALID_SMOKING),
                index=2,
                format_func=lambda v: format_category_pt("smoking_status", v),
            )
            alcohol_consumption = st.select_slider(
                "Consumo de Álcool",
                list(VALID_LEVEL),
                value="Medium",
                format_func=lambda v: format_category_pt("alcohol_consumption", v),
            )
            diet_type = st.selectbox(
                "Padrão Alimentar",
                list(VALID_DIET),
                index=2,
                format_func=lambda v: format_category_pt("diet_type", v),
            )
            red_meat_consumption = st.select_slider(
                "Consumo de Carne Vermelha",
                list(VALID_LEVEL),
                value="Medium",
                format_func=lambda v: format_category_pt("red_meat_consumption", v),
            )
            fiber_consumption = st.select_slider(
                "Consumo de Fibra",
                list(VALID_LEVEL),
                value="Low",
                format_func=lambda v: format_category_pt("fiber_consumption", v),
            )

        submit_btn = st.form_submit_button("🩺 Avaliar Doente", width="stretch")

    raw_payload: Dict[str, Any] = dict(
        age=age,
        bmi=bmi,
        gender=gender,
        race=race,
        region=region,
        urban_or_rural=urban_or_rural,
        socioeconomic_status=socioeconomic_status,
        insurance_coverage=insurance_coverage,
        family_history=family_history,
        previous_cancer_history=previous_cancer_history,
        stage_at_diagnosis=stage_at_diagnosis,
        tumor_aggressiveness=tumor_aggressiveness,
        colonoscopy_access=colonoscopy_access,
        time_to_diagnosis=time_to_diagnosis,
        treatment_access=treatment_access,
        surgery_received=surgery_received,
        chemotherapy_received=chemotherapy_received,
        radiotherapy_received=radiotherapy_received,
        follow_up_adherence=follow_up_adherence,
        screening_regularity=screening_regularity,
        diet_type=diet_type,
        physical_activity_level=physical_activity_level,
        smoking_status=smoking_status,
        alcohol_consumption=alcohol_consumption,
        red_meat_consumption=red_meat_consumption,
        fiber_consumption=fiber_consumption,
    )

    cached_patient = st.session_state.get("validated_patient")
    schema_changed = cached_patient is None or not hasattr(cached_patient, "stage_at_diagnosis")

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
    """Intervenções modificáveis selecionadas pelo clínico (cenários contrafactuais)."""
    st.sidebar.divider()
    st.sidebar.header("🔮 Simulação de Cenários")
    applicable = [k for k in INTERVENTIONS if is_intervention_applicable(patient, k)]
    if not applicable:
        st.sidebar.info("Todas as 8 metas modificáveis já foram atingidas.")
        return []
    if "sim_choice" in st.session_state:
        st.session_state["sim_choice"] = [k for k in st.session_state["sim_choice"] if k in applicable]
    return st.sidebar.multiselect(
        "Intervenções a simular",
        applicable,
        format_func=lambda k: INTERVENTIONS[k][0],
        key="sim_choice",
    )


def _horizon_label(h: int) -> str:
    if h == 12:
        return "Livre de recidiva a 1 ano (12m)"
    if h == 36:
        return "Livre de recidiva a 3 anos (36m)"
    return f"Livre de recidiva aos {h}m (máx. coorte)"


# ============================================================================
# CONTROLADOR PRINCIPAL
# ============================================================================
def main() -> None:
    st.title("🧬 Prognóstico e Sobrevida Livre de Recidiva — Cancro Colorretal")
    st.caption(
        "Modelo de sobrevida livre de recidiva até 59 meses "
        "(variáveis pós-desfecho excluídas para prevenir fuga de informação)."
    )

    model = get_model()
    bundle = get_bundle_from_disk()

    if model is None:
        st.error(
            f"Modelo não encontrado em `{MODEL_PATH}`.\n\n"
            "**Porquê:** a aplicação opera em 'Separation of Concerns' (apenas inferência); "
            "necessita do artefacto `cox_survival_model.pkl` previamente treinado.\n\n"
            "**Solução:** carregue o ficheiro abaixo ou execute `python scripts/export_notebook_models.py`."
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
            c2.metric("Estádio ao Diagnóstico", validated_patient.stage_at_diagnosis)
            c3.metric(
                "Agressividade",
                format_category_pt("tumor_aggressiveness", validated_patient.tumor_aggressiveness),
            )
            c4.metric("IMC", f"{validated_patient.bmi:.1f} kg/m²")
        else:
            st.error("⛔ Dados com inconformidade clínica. A inferência está bloqueada para prevenir predições enviesadas.")

        st.caption(
            f"Limites aceites na entrada: idade [{LIMITS['age'][0]:.0f}–{LIMITS['age'][1]:.0f} anos], "
            f"IMC [{LIMITS['bmi'][0]:.1f}–{LIMITS['bmi'][1]:.1f} kg/m²] | "
            f"Winsorização do treino (5%–95%): idade [{WINSOR_BOUNDS['Age'][0]:.0f}–{WINSOR_BOUNDS['Age'][1]:.0f}], "
            f"IMC [{WINSOR_BOUNDS['BMI'][0]:.1f}–{WINSOR_BOUNDS['BMI'][1]:.1f}]."
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
            render_error("Falha ao calcular a curva de sobrevida livre de recidiva.", exc)
        return

    with tab_pred:
        if bundle is not None:
            st.subheader("Classificação de Risco de Recidiva (LightGBM + Calibração de Platt)")
            clf_res = predict_recurrence_risk(patient_dict, bundle)
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                render_metric_card("Risco calibrado (Platt)", clf_res["prob_calibrated"], invert_color=True)
            with rc2:
                render_metric_card(
                    f"Pontuação bruta (limiar = {clf_res['threshold']:.4f})",
                    clf_res["prob_raw"],
                    invert_color=True,
                )
            with rc3:
                status_txt = "Alto Risco (≥ Limiar Clínico)" if clf_res["high_risk"] else "Baixo Risco (< Limiar Clínico)"
                st.metric("Decisão pelo Limiar Clínico", status_txt)
            st.divider()

        st.subheader("Projeção Temporal Livre de Recidiva (Modelo CoxPH)")
        cols = st.columns(len(HORIZONS_MONTHS))
        for col, h in zip(cols, HORIZONS_MONTHS):
            with col:
                render_metric_card(_horizon_label(h), surv[h])

        st.subheader("Curvas livres de recidiva estratificadas pela mediana")
        try:
            strata = get_strata(model, tuple(sorted(patient_dict.items())), patient_dict)
            st.info(f"O doente pertence ao estrato (mediana da coorte): **{strata['patient_group']}**.")
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
        sub_tabs = ["SHAP — LightGBM (TreeExplainer)", "SHAP — CoxPH (26 variáveis)", "LIME — CoxPH (26 variáveis)"]
        sub_lgb, sub_shap, sub_lime = st.tabs(sub_tabs)

        with sub_lgb:
            if bundle is None:
                st.warning("Bundle `clinical_pipeline_bundle.pkl` não encontrado; explicabilidade LightGBM indisponível.")
            else:
                st.caption(
                    "Mesmo `TreeExplainer` utilizado na Fase 5.4 do Notebook sobre o `LightGBM (Otimizado)` "
                    "(valores positivos empurram para maior probabilidade de recidiva)."
                )
                try:
                    lgb_exp = get_lgb_explainer(bundle, id(bundle), SCHEMA_VERSION)
                    with st.spinner("A calcular valores SHAP (`TreeExplainer`)…"):
                        fig_lgb = generate_lgb_shap_plot(patient_dict, bundle, lgb_exp)
                    st.pyplot(fig_lgb)
                except Exception as exc:  # noqa: BLE001
                    render_error("Não foi possível gerar a explicação SHAP do LightGBM.", exc)

        with sub_shap:
            st.caption("Valores positivos aumentam o log-hazard do `CoxPH`; valores negativos reduzem o risco.")
            try:
                explainer = get_shap_explainer(model, id(model), SCHEMA_VERSION)
                with st.spinner("A calcular valores SHAP locais (CoxPH)…"):
                    fig = generate_shap_plot(patient_dict, model, explainer)
                st.pyplot(fig)
            except Exception as exc:  # noqa: BLE001
                render_error("Não foi possível gerar a explicação SHAP do CoxPH.", exc)

        with sub_lime:
            st.caption("O LIME ajusta um modelo linear interpretável em torno deste doente nas 26 covariáveis reais.")
            try:
                lime_exp = get_lime_explainer(model, id(model), SCHEMA_VERSION)
                with st.spinner("A calcular explicação LIME local…"):
                    lfig, ldf = generate_lime_plot(patient_dict, model, lime_exp)
                st.pyplot(lfig)
                with st.expander("Tabela de pesos locais LIME"):
                    st.dataframe(ldf, width="stretch", hide_index=True)
            except Exception as exc:  # noqa: BLE001
                render_error("Não foi possível gerar a explicação LIME.", exc)

    with tab_sim:
        st.subheader("Simulador Contrafactual (8 Fatores Modificáveis)")
        st.caption(
            "Apenas os fatores selecionados são alterados de forma independente; "
            "as restantes covariáveis mantêm-se fixas."
        )
        options = [k for k in INTERVENTIONS if is_intervention_applicable(patient_dict, k)]
        if not options:
            st.info("Todas as 8 intervenções modificáveis já estão aplicadas.")
        else:
            labels = [INTERVENTIONS[k][0] for k in options]
            values = [
                risk_reduction_pct(patient_dict, apply_interventions(patient_dict, [k]), model)
                for k in options
            ]
            highlight = None
            if chosen:
                combo_label = "Combinação selecionada"
                scenario = apply_interventions(patient_dict, chosen)
                cf_res = counterfactual_summary(patient_dict, scenario, model, bundle)
                labels.append(combo_label)
                values.append(cf_res["cox_reduction_pct"])
                highlight = combo_label

                sim_curve = predict_survival_curve(scenario, model)
                sim = survival_at(sim_curve)
                cols = st.columns(len(HORIZONS_MONTHS))
                for col, h in zip(cols, HORIZONS_MONTHS):
                    with col:
                        render_metric_card(f"{_horizon_label(h)} — simulado", sim[h], sim[h] - surv[h])

                if bundle is not None:
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Redução de Risco (CoxPH)", f"{cf_res['cox_reduction_pct']:+.2f}%")
                    k2.metric("Redução Relativa (LightGBM Calibrado)", f"{cf_res['lgb_cal_reduction_pct']:+.2f}%")
                    k3.metric("Redução Relativa (LightGBM Pontuação Bruta)", f"{cf_res['lgb_raw_reduction_pct']:+.2f}%")

                st.plotly_chart(survival_figure(curve, sim_curve), width="stretch")
            else:
                st.info("Escolha intervenções na barra lateral para projetar a curva simulada.")

            st.markdown("#### Redução percentual do risco por intervenção (CoxPH)")
            st.plotly_chart(risk_reduction_figure(labels, values, highlight), width="stretch")


if __name__ == "__main__":
    main()
