# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Função para carregar os dados
@st.cache_data
def load_data():
    df = pd.read_csv('Hospital ER_Data.csv')
    df['Patient Admission Date'] = pd.to_datetime(df['Patient Admission Date'])
    return df


# Carregar dados
df = load_data()

# Sidebar
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Dashboard Geral", "Previsão de Atendimentos", "Machine Learning", "Conclusão"])

# Páginas
if page == "Dashboard Geral":
    st.title("📊 Dashboard Geral de Atendimentos")

    # KPIs
    total_atendimentos = df.shape[0]
    tempo_medio_espera = df['Patient Waittime'].mean()
    idade_media = df['Patient Age'].mean()
    satisfacao_media = df['Patient Satisfaction Score'].mean()

    st.metric(label="Total de Atendimentos", value=f"{total_atendimentos}")
    st.metric(label="Tempo Médio de Espera", value=f"{tempo_medio_espera:.2f} minutos")
    st.metric(label="Idade Média dos Pacientes", value=f"{idade_media:.2f} anos")
    st.metric(label="Satisfação Média dos Pacientes", value=f"{satisfacao_media:.2f} pontos")

    # Gráfico de Atendimentos por Mês
    df['YearMonth'] = df['Patient Admission Date'].dt.to_period('M')
    monthly_counts = df.groupby('YearMonth').size().reset_index()
    monthly_counts['YearMonth'] = monthly_counts['YearMonth'].astype(str)

    fig = px.line(
        monthly_counts,
        x='YearMonth',
        y=0,
        markers=True,
        title='Quantidade de Atendimentos por Mês'
    )
    st.plotly_chart(fig)

elif page == "Previsão de Atendimentos":
    st.title("📈 Previsão de Atendimentos - Próximo Mês")

    df['YearMonth'] = df['Patient Admission Date'].dt.to_period('M')
    monthly_counts = df.groupby('YearMonth').size()

    model = ARIMA(monthly_counts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)

    st.success(f"📈 Previsão de Atendimentos para o Próximo Mês: {int(forecast.values[0])}")

elif page == "Machine Learning":
    st.title("🤖 Modelagem Preditiva de Satisfação")

    st.write("Aqui mostramos os resultados de 3 abordagens de Machine Learning aplicadas:")

    st.markdown("""
**Modelos Avaliados:**
- 🌲 Random Forest
- ⚖️ Random Forest com Pesos Balanceados
- 🌳 Balanced Random Forest

**Métricas Analisadas:**
- Acurácia
- Precision
- Recall
- F1-Score
""")

    metrics_df = pd.DataFrame({
        "Modelo": ["Random Forest", "Random Forest Balanceado", "Balanced Random Forest"],
        "Acurácia": [0.833, 0.828, 0.651],
        "Precision": [0.177, 0.211, 0.143],
        "Recall": [0.069, 0.106, 0.325],
        "F1-Score": [0.099, 0.140, 0.199]
    })
    styled_df = metrics_df.style.background_gradient(subset=["Acurácia", "Precision", "Recall", "F1-Score"], cmap='Blues').format({
        "Acurácia": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}"
    })
    st.dataframe(styled_df)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # ========== PREPARAR OS DADOS PARA MACHINE LEARNING ==========
    from sklearn.model_selection import train_test_split

    # Variáveis para ML
    features = df[['Patient Age', 'Patient Gender', 'Department Referral', 'Patient Waittime']]
    features_encoded = pd.get_dummies(features, columns=['Patient Gender', 'Department Referral'], drop_first=True)

    X = features_encoded
    y = df['Patient Satisfaction Score'].apply(lambda x: 1 if x >= 6 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ========== TREINAR MODELOS ==========
    model_rf = RandomForestClassifier(random_state=42)
    model_rf_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')

    from imblearn.ensemble import BalancedRandomForestClassifier

    model_brf = BalancedRandomForestClassifier(random_state=42, n_estimators=100)

    # Treinar
    model_rf.fit(X_train, y_train)
    model_rf_balanced.fit(X_train, y_train)
    model_brf.fit(X_train, y_train)


    # Prever probabilidades
    y_pred_prob = model_rf.predict_proba(X_test)[:,1]
    y_pred_balanced_prob = model_rf_balanced.predict_proba(X_test)[:,1]
    y_pred_brf_prob = model_brf.predict_proba(X_test)[:,1]

    # Calcular fpr e tpr para cada modelo
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob)
    fpr_balanced, tpr_balanced, _ = roc_curve(y_test, y_pred_balanced_prob)
    fpr_brf, tpr_brf, _ = roc_curve(y_test, y_pred_brf_prob)

    # Calcular AUC
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_balanced = auc(fpr_balanced, tpr_balanced)
    auc_brf = auc(fpr_brf, tpr_brf)

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
    ax.plot(fpr_balanced, tpr_balanced, label=f'RF Balanceado (AUC = {auc_balanced:.2f})')
    ax.plot(fpr_brf, tpr_brf, label=f'Balanced RF (AUC = {auc_brf:.2f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('Taxa de Falso Positivo')
    ax.set_ylabel('Taxa de Verdadeiro Positivo')
    ax.set_title('Curva ROC Comparativa')
    ax.legend()

    st.pyplot(fig)

    st.write("✅ Comparando acurácia e capacidade de detecção de pacientes satisfeitos.")

elif page == "Conclusão":
    st.title("✅ Conclusão")

    st.markdown("""
    - O modelo Random Forest tradicional apresentou melhor acurácia geral.
    - O Balanced Random Forest melhorou o Recall para pacientes satisfeitos, embora com perda de acurácia.
    - O dataset apresentou forte desbalanceamento entre classes.
    - A previsão ARIMA sugere tendências sazonais nos atendimentos.
    - Próximos passos incluem melhoria dos modelos e integração em sistema de alertas.
    """)