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
st.sidebar.header("Filtros Interactivos")

clubs = st.sidebar.multiselect("Club", options=sorted(df['Club'].unique()))
nationalities = st.sidebar.multiselect("Nacionalidad Principal", options=sorted(df['Primary Nationality'].unique()))
positions = st.sidebar.multiselect("Posición", options=sorted(df['Position'].unique()))

min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Rango de Edad", min_age, max_age, (min_age, max_age))

# Usar listas de Python para evitar errores de tipo de numpy
df_filtered_pre = df[
    (df['Age'].between(age_range[0], age_range[1]))
]
if clubs:
    df_filtered_pre = df_filtered_pre[df_filtered_pre['Club'].isin(clubs)]
if nationalities:
    df_filtered_pre = df_filtered_pre[df_filtered_pre['Primary Nationality'].isin(nationalities)]
if positions:
    df_filtered_pre = df_filtered_pre[df_filtered_pre['Position'].isin(positions)]

# --- Cuerpo Principal de la App ---
st.title("📊 Dashboard Interactivo de Scouting")
st.markdown(f"Mostrando **{len(df_filtered_pre)}** de **{len(df)}** jugadores según los filtros seleccionados.")

# --- Pestañas para organizar el contenido ---
tab1, tab2, tab3, tab4 = st.tabs(["Visión General", "Análisis de Rendimiento", "Análisis Financiero", "🤖 Agente IA"])

with tab1:
    st.header("Visión General de los Datos Seleccionados")
    st.dataframe(df_filtered_pre)
    st.header("Correlación de Métricas")
    st.pyplot(plot_correlation_heatmap(df_filtered_pre))

with tab2:
    st.header("Análisis de Rendimiento")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_top_players(df_filtered_pre, 'Goals', 'Top 10 Goleadores'))
    with col2:
        st.pyplot(plot_top_players(df_filtered_pre, 'Assists', 'Top 10 Asistidores'))
    st.pyplot(plot_top_players(df_filtered_pre, 'Performance', 'Top 10 por Rendimiento Total'))

with tab3:
    st.header("Análisis Financiero y de Eficiencia")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_value_distribution(df_filtered_pre))
    with col2:
        st.pyplot(plot_top_players(df_filtered_pre, 'Market Value', 'Top 10 Jugadores más Valiosos'))
    st.header("Análisis de Eficiencia (Moneyball)")
    st.pyplot(plot_efficiency_scatter(df_filtered_pre))

with tab4:
    st.header("Asistente de Scouting con IA")
    st.info("El agente analizará el conjunto de datos **filtrado actualmente** para darte recomendaciones específicas.")
    
    api_key_input = st.text_input("Introduce tu API Key de Groq", type="password", value=GROQ_API_KEY or "")
    
    if not api_key_input:
        st.warning("Se necesita una API Key de Groq para usar el agente.")
    else:
        summary = get_dynamic_eda_summary(df_filtered_pre)
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
