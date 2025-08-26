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

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Dashboard de Scouting")
st.title("📊 Dashboard Interactivo de Scouting")

# Cargar API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Función de Carga de Datos Cacheada ---
# Se ejecuta solo cuando el archivo subido cambia, optimizando el rendimiento.
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df_featured = create_features(df)
    return df_featured

# --- Sección para Subir el Archivo ---
uploaded_file = st.file_uploader(
    "Sube tu archivo CSV con estadísticas de jugadores para comenzar", 
    type="csv"
)

# --- Lógica Principal de la Aplicación ---
# Todo el dashboard se construye solo si un archivo ha sido subido.
if uploaded_file is not None:
    df = load_data(uploaded_file)

    # --- Sidebar de Filtros (se crea dinámicamente) ---
    st.sidebar.header("Filtros Interactivos")

    clubs = st.sidebar.multiselect("Club", options=sorted(df['Club'].unique()))
    nationalities = st.sidebar.multiselect("Nacionalidad Principal", options=sorted(df['Primary Nationality'].unique()))
    positions = st.sidebar.multiselect("Posición", options=sorted(df['Position'].unique()))

    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.sidebar.slider("Rango de Edad", min_age, max_age, (min_age, max_age))

    # Aplicar filtros al DataFrame
    df_filtered = df.copy()
    if clubs:
        df_filtered = df_filtered[df_filtered['Club'].isin(clubs)]
    if nationalities:
        df_filtered = df_filtered[df_filtered['Primary Nationality'].isin(nationalities)]
    if positions:
        df_filtered = df_filtered[df_filtered['Position'].isin(positions)]
    df_filtered = df_filtered[df_filtered['Age'].between(age_range[0], age_range[1])]


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

else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis.")
