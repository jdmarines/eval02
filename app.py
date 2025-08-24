# app.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Importar funciones de nuestros módulos
from eda import (
    create_features, 
    plot_correlation_heatmap, 
    plot_value_distribution, 
    plot_top_players, 
    plot_efficiency_scatter,
    get_dynamic_eda_summary
)
from agent import get_agent_response

# --- Configuración de la Página y Carga de Datos ---
st.set_page_config(layout="wide", page_title="Dashboard de Scouting")

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df_featured = create_features(df)
    return df_featured

# Cargar API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Cargar el dataset
try:
    df = load_data('Top 500 Players 2024.csv')
except FileNotFoundError:
    st.error("Error: El archivo 'Top 500 Players 2024.csv' no se encontró. Asegúrate de que esté en la misma carpeta que app.py.")
    st.stop()


# --- Sidebar de Filtros ---
st.sidebar.header("Filtros Interactivos  фильтры")

clubs = st.sidebar.multiselect("Club", options=df['Club'].unique(), default=df['Club'].unique())
nationalities = st.sidebar.multiselect("Nacionalidad Principal", options=df['Primary Nationality'].unique(), default=df['Primary Nationality'].unique())
positions = st.sidebar.multiselect("Posición", options=df['Position'].unique(), default=df['Position'].unique())

min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Rango de Edad", min_age, max_age, (min_age, max_age))

min_value, max_value = int(df['Market Value'].min()), int(df['Market Value'].max())
value_range = st.sidebar.slider("Rango de Valor de Mercado (EUR)", min_value, max_value, (min_value, max_value))

# Aplicar filtros al DataFrame
df_filtered = df[
    (df['Club'].isin(clubs)) &
    (df['Primary Nationality'].isin(nationalities)) &
    (df['Position'].isin(positions)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Market Value'].between(value_range[0], value_range[1]))
]

# --- Cuerpo Principal de la App ---
st.title("📊 Dashboard Interactivo de Scouting")
st.markdown(f"Mostrando **{len(df_filtered)}** de **{len(df)}** jugadores según los filtros seleccionados.")

# --- Pestañas para organizar el contenido ---
tab1, tab2, tab3, tab4 = st.tabs(["Visión General", "Análisis de Rendimiento", "Análisis Financiero", "🤖 Agente IA"])

with tab1:
    st.header("Visión General de los Datos Seleccionados")
    st.dataframe(df_filtered)
    st.header("Correlación de Métricas")
    st.pyplot(plot_correlation_heatmap(df_filtered))

with tab2:
    st.header("Análisis de Rendimiento")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_top_players(df_filtered, 'Goals', 'Top 10 Goleadores'))
    with col2:
        st.pyplot(plot_top_players(df_filtered, 'Assists', 'Top 10 Asistidores'))
    st.pyplot(plot_top_players(df_filtered, 'Performance', 'Top 10 por Rendimiento Total'))

with tab3:
    st.header("Análisis Financiero y de Eficiencia")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_value_distribution(df_filtered))
    with col2:
        st.pyplot(plot_top_players(df_filtered, 'Market Value', 'Top 10 Jugadores más Valiosos'))
    st.header("Análisis de Eficiencia (Moneyball)")
    st.pyplot(plot_efficiency_scatter(df_filtered))

with tab4:
    st.header("Asistente de Scouting con IA")
    st.info("El agente analizará el conjunto de datos **filtrado actualmente** para darte recomendaciones específicas.")
    
    api_key_input = st.text_input("Introduce tu API Key de Groq", type="password", value=GROQ_API_KEY or "")
    
    if not api_key_input:
        st.warning("Se necesita una API Key de Groq para usar el agente.")
    else:
        # Generar resumen dinámico
        summary = get_dynamic_eda_summary(df_filtered)
        st.markdown("#### Resumen para el Agente:")
        with st.expander("Ver el resumen que recibirá la IA"):
            st.text(summary)

        user_question = st.text_area("Haz una pregunta específica sobre los jugadores seleccionados:", height=100)
        
        if st.button("Consultar al Agente"):
            if user_question:
                with st.spinner("El Director Deportivo está analizando los datos..."):
                    response = get_agent_response(api_key_input, summary, user_question)
                    st.success(response)
            else:
                st.warning("Por favor, introduce una pregunta.")
