# Pipeline Clínico de Machine Learning e Análise de Sobrevida — Cancro Colorretal (V2)

Repositório estruturado sob o princípio de **Separation of Concerns**:
- **Treino e Validação Científica ([`Colorectal_Cancer_Prognosis_V2.ipynb`](./Colorectal_Cancer_Prognosis_V2.ipynb))**: Executa o protocolo completo em 5 Fases (Zero Data Leakage, amostragem híbrida `RENN + SMOTE` *In-Fold*, otimização de hiperparâmetros com CV=5, calibração de Platt OOF, sobrevivência temporal CoxPH, 1.000 iterações de *bootstrapping*, curvas DCA e simulação contrafactual) e exporta apenas o modelo vencedor e as melhores métricas.
- **Interface Clínica de Decisão ([`app.py`](./app.py))**: Aplicação web em Streamlit focada exclusivamente em inferência clínica limpa (validação Pydantic v2, previsão de sobrevivência a 1, 3 e 5 anos, explicabilidade local SHAP/LIME e simulador contrafactual *What-If*).

---

## 1. Estrutura de Pastas e Ficheiros

```text
Colorectal_Cancer_Prognosis_V2/
├── .streamlit/
│   └── config.toml                        # Configuração de tema da interface clínica Streamlit
├── data/
│   └── colorectal_cancer_prediction.csv   # Dataset clínico (89.945 registos brutos -> 88.494 válidos após QMP)
├── models/
│   ├── cox_survival_model.pkl             # Modelo vencedor de sobrevivência (CoxPH L2=0.5) exportado do Notebook
│   ├── clinical_pipeline_bundle.pkl       # Classificador vencedor (LightGBM + Platt OOF) e métricas oficiais validadas
│   └── reference_cohort.csv               # Coorte real de referência (1.000 doentes do Treino)
├── artifacts/
│   ├── processed_splits_v2.pkl            # Splits 80/20 + RENN+SMOTE Global e In-Fold (Fase 2)
│   ├── phase3_results_v2.pkl              # Modelos otimizados (CV=5), OOF, Calibrador Platt e Limiares (Fase 3)
│   ├── phase4_results_v2.pkl              # Resultados de sobrevivência temporal e sensibilidade a 12m (Fase 4)
│   └── metricas_finais_protocolo_v2.csv   # Tabela oficial consolidada (Teste + IC 95% de 1.000 Bootstraps)
├── plots/
│   ├── phase1_qmp_overview.png            # Fase 1: QMP e Auditoria de Sinal Univariado
│   ├── phase2_class_balance.png           # Fase 2: Balanceamento RENN + SMOTE vs. Teste Intocado
│   ├── phase3_threshold_and_cm.png        # Fase 3: Curva PR OOF, Threshold Moving e Matriz de Confusão
│   ├── phase4_cox_hazard_ratios.png       # Fase 4: Forest Plot CoxPH e Curvas de Kaplan-Meier (12m)
│   ├── phase5_dca_and_counterfactual.png  # Fase 5: Decision Curve Analysis (DCA) e Contrafactual What-If
│   ├── shap_summary_beeswarm.png          # Fase 5: SHAP Beeswarm Plot (Top 10 Fatores)
│   └── shap_summary_bar.png               # Fase 5: SHAP Bar Plot (Média |SHAP|)
├── src/
│   ├── __init__.py
│   ├── data_validation.py                 # Validação clínica Pydantic v2 (Single-Batch Validation)
│   ├── inference.py                       # Motor de inferência clínica (curvas de sobrevivência, estratos e What-If)
│   ├── explainability.py                  # Explicabilidade local SHAP Waterfall e LIME nas variáveis clínicas
│   └── ui_components.py                   # Componentes gráficos Plotly e cartões de decisão clínica
├── scripts/
│   ├── export_notebook_models.py          # Exporta o modelo vencedor e as melhores métricas para `models/`
│   └── make_demo_model.py                 # Alias compatível que delega para `export_notebook_models.py`
├── Colorectal_Cancer_Prognosis_V2.ipynb   # Notebook de treino e validação (Fases 1 a 5 executadas)
├── app.py                                 # Aplicação Clínica Streamlit (Visão Geral, Previsão, Explicabilidade, Simulação)
├── requirements.txt                       # Dependências do projeto
└── README.md                              # Documentação geral
```

---

## 2. Como Executar a Interface Clínica

1. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Sincronizar o modelo vencedor do Notebook para `models/` (já executado)**:
   ```bash
   python scripts/export_notebook_models.py
   ```
3. **Iniciar a Interface Web Streamlit**:
   ```bash
   streamlit run app.py
   ```

---

## 3. Melhores Métricas Extraídas do Notebook e Integradas na Interface

| Métrica Validada no Conjunto de Teste ($N = 17.699$) | Modelo Campeão | Resultado Obtido (IC 95% — 1.000 Bootstraps) |
| :--- | :---: | :---: |
| **Discriminação Temporal (`C-Index`)** | `CoxPH (L2 = 0.5)` | `0.5097 [0.5005, 0.5186]` |
| **Sensibilidade Temporal a 1 Ano ($t \le 12\text{m}$)** | `LightGBM + Limiar Clínico OOF (0.5082)` | `82.85% [80.58%, 85.24%]` |
| **Calibração de Probabilidades (`Brier Score`)** | `LightGBM + Calibração Logística Platt OOF` | `0.2100 [0.2072, 0.2128]` |
| **Sensibilidade Global (`Recall Classe 1`)** | `LightGBM + Limiar Clínico OOF (0.5082)` | `82.34% [81.25%, 83.32%]` |
| **F1-Macro (Equilíbrio entre Classes)** | `LightGBM + Limiar Youden OOF (0.5678)` | `0.4904` *(ou `0.3606` no limiar de alta sensibilidade)* |
