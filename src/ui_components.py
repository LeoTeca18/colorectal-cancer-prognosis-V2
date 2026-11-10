"""Reusable Streamlit components."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CSS = """
<style>
.metric-card{background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;
border-radius:14px;padding:20px;text-align:center;box-shadow:0 4px 14px rgba(0,0,0,.3)}
.metric-card .label{color:#94a3b8;font-size:.9rem;letter-spacing:.05em;text-transform:uppercase}
.metric-card .value{font-size:2.4rem;font-weight:700;margin:6px 0}
.metric-card .delta{font-size:.9rem;color:#34d399}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_metric_card(label: str, probability: float, delta: Optional[float] = None) -> None:
    """Card showing a survival probability (0-1) with optional delta vs. baseline."""
    color = "#34d399" if probability >= .75 else "#fbbf24" if probability >= .5 else "#f87171"
    d = ""
    if delta is not None:
        d = f'<div class="delta">{delta * 100:+.1f} p.p. vs. cenário base</div>'
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{probability * 100:.1f}%</div>{d}</div>',
                unsafe_allow_html=True)


def survival_figure(curve: pd.DataFrame, scenario: Optional[pd.DataFrame] = None,
                    scenario_label: str = "Cenário simulado") -> go.Figure:
    """Survival curve over months, optionally overlaying a counterfactual."""
    fig = go.Figure()
    fig.add_scatter(x=curve["month"], y=curve["survival"], mode="lines",
                    name="Cenário atual", line=dict(color="#38bdf8", width=3, shape="hv"))
    if scenario is not None:
        fig.add_scatter(x=scenario["month"], y=scenario["survival"], mode="lines",
                        name=scenario_label, line=dict(color="#34d399", width=3, dash="dash", shape="hv"),
                        fill="tonexty", fillcolor="rgba(52,211,153,.12)")
    for h in (12, 36, 60):
        fig.add_vline(x=h, line_dash="dot", line_color="#64748b")
    fig.update_layout(template="plotly_dark", height=420, xaxis_title="Meses",
                      yaxis_title="Probabilidade de sobrevivência",
                      yaxis=dict(range=[0, 1.02], tickformat=".0%"),
                      legend=dict(orientation="h", y=1.1))
    return fig


STRATA_COLORS = {"Baixo risco": "#34d399", "Risco intermédio": "#fbbf24", "Alto risco": "#f87171"}


def strata_figure(strata: dict, patient_curve: pd.DataFrame) -> go.Figure:
    """Stratified Kaplan-Meier-style curves (low/medium/high risk) plus the patient."""
    fig = go.Figure()
    for name, df in strata["curves"].items():
        fig.add_scatter(x=df["month"], y=df["survival"], mode="lines", name=name,
                        line=dict(color=STRATA_COLORS[name], width=2, shape="hv"),
                        hovertemplate="%{x} meses: %{y:.1%}<extra>" + name + "</extra>")
    fig.add_scatter(x=patient_curve["month"], y=patient_curve["survival"], mode="lines",
                    name="Doente", line=dict(color="#38bdf8", width=4, shape="hv"),
                    hovertemplate="%{x} meses: %{y:.1%}<extra>Doente</extra>")
    for h in (12, 36, 60):
        fig.add_vline(x=h, line_dash="dot", line_color="#64748b",
                      annotation_text=f"{h // 12}a", annotation_position="top")
    fig.update_layout(template="plotly_dark", height=450, xaxis_title="Meses",
                      yaxis_title="Probabilidade de sobrevivência",
                      yaxis=dict(range=[0, 1.02], tickformat=".0%"),
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return fig


def risk_reduction_figure(labels: list, values: list, highlight: Optional[str] = None) -> go.Figure:
    """Horizontal bar chart of % hazard reduction per simulated intervention."""
    colors = ["#38bdf8" if l == highlight else "#34d399" if v > 0 else "#64748b"
              for l, v in zip(labels, values)]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:.1f}%" for v in values], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=80 + 55 * len(labels),
                      xaxis_title="Redução relativa do risco (%)", yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=40, t=20, b=10))
    return fig


def explain_exception(exc: Exception) -> str:
    """Translate a technical exception into a likely cause and a suggested fix."""
    msg = str(exc)
    if isinstance(exc, KeyError) or "not in index" in msg or "columns" in msg.lower():
        return ("Causa provável: as colunas esperadas pelo modelo não coincidem com as usadas "
                "pela aplicação (lista `FEATURE_COLUMNS` em `src/inference.py`). "
                "Solução: alinhe essa lista com as variáveis do treino.")
    if isinstance(exc, AttributeError):
        return ("Causa provável: o ficheiro .pkl não é um modelo `lifelines.CoxPHFitter` (ou tem "
                "outra versão) e não expõe o método pedido. Solução: exporte o modelo como "
                "CoxPHFitter ou adapte `src/inference.py` à interface do seu modelo.")
    if isinstance(exc, (ValueError, TypeError)):
        return ("Causa provável: os dados enviados ao modelo têm formato ou valores que ele não "
                "aceita (por exemplo NaN, tipo errado ou número de colunas diferente). "
                "Solução: confirme os campos da barra lateral e a coorte de referência.")
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return ("Causa provável: falta uma biblioteca no ambiente Python. "
                "Solução: execute `pip install -r requirements.txt`.")
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return ("Causa provável: um ficheiro necessário não existe ou não pode ser lido. "
                "Solução: verifique a pasta `models/` e as permissões.")
    return "Causa desconhecida: consulte os detalhes técnicos abaixo."


def render_error(title: str, exc: Exception, warning: bool = False) -> None:
    """Show an error/warning with the reason, a suggested fix and the technical detail."""
    box = st.warning if warning else st.error
    box(f"**{title}**\n\n{explain_exception(exc)}")
    with st.expander("Detalhes técnicos"):
        st.code(f"{type(exc).__name__}: {exc}")
