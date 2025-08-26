# app.py
# --------------------------------------------------
# Archivo principal de la aplicación.
# Aquí se construye el dashboard interactivo con Streamlit,
# se cargan los datos, se aplican filtros, se generan gráficos,
# y se integra el agente de IA para consultas en lenguaje natural.
# --------------------------------------------------

# --- Importaciones principales ---
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# --- Importar funciones personalizadas de los módulos ---
from eda import (
    create_features,                 # Feature engineering (nuevas columnas)
    plot_correlation_heatmap,        # Gráfico de correlación entre variables
    plot_value_distribution,         # Distribución de valores de mercado
    plot_top_players,                # Ranking de jugadores según métrica
    plot_efficiency_scatter,         # Scatter tipo "Moneyball"
    get_dynamic_eda_summary          # Resumen dinámico del dataset filtrado
)
from agent import get_agent_response # Función para obtener respuesta del LLM


# --- Configuración de la Página en Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard de Scouting")
st.title("📊 Dashboard Interactivo de Scouting")


# --- Cargar la API Key desde variables de entorno ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --- Función cacheada para cargar y preparar datos ---
@st.cache_data
def load_data(uploaded_file):
    """
    Carga los datos desde un CSV y aplica feature engineering.
    Gracias a @st.cache_data, esta función solo se ejecuta de nuevo 
    si cambia el archivo subido → mejora el rendimiento.
    """
    df = pd.read_csv(uploaded_file)
    df_featured = create_features(df)
    return df_featured


# --- Sección para subir un archivo CSV ---
uploaded_file = st.file_uploader(
    "Sube tu archivo CSV con estadísticas de jugadores para comenzar", 
    type="csv"
)


# --- Lógica Principal de la Aplicación ---
if uploaded_file is not None:
    # 1. Cargar y preparar datos
    df = load_data(uploaded_file)

    # 2. Sidebar de Filtros
    st.sidebar.header("Filtros Interactivos")

    clubs = st.sidebar.multiselect("Club", options=sorted(df['Club'].unique()))
    nationalities = st.sidebar.multiselect("Nacionalidad Principal", options=sorted(df['Primary Nationality'].unique()))
    positions = st.sidebar.multiselect("Posición", options=sorted(df['Position'].unique()))

    # Rango dinámico de edad
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.sidebar.slider("Rango de Edad", min_age, max_age, (min_age, max_age))

    # Aplicación de filtros al DataFrame
    df_filtered = df.copy()
    if clubs:
        df_filtered = df_filtered[df_filtered['Club'].isin(clubs)]
    if nationalities:
        df_filtered = df_filtered[df_filtered['Primary Nationality'].isin(nationalities)]
    if positions:
        df_filtered = df_filtered[df_filtered['Position'].isin(positions)]
    df_filtered = df_filtered[df_filtered['Age'].between(age_range[0], age_range[1])]

    # Info resumen de jugadores filtrados
    st.markdown(f"Mostrando **{len(df_filtered)}** de **{len(df)}** jugadores según los filtros seleccionados.")


    # 3. Pestañas principales del dashboard
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Agente IA"
        "Visión General", 
        "Análisis de Rendimiento", 
        "Análisis Financiero", 
    ])

    
    # --- TAB 1: Agente IA ---
    with tab1:
        st.header("Asistente de Scouting con IA")
        st.info("El agente analizará el conjunto de datos **filtrado actualmente** para darte recomendaciones específicas.")
        
        # Entrada de API Key
        api_key_input = st.text_input("Introduce tu API Key de Groq", type="password", value=GROQ_API_KEY or "")
        
        if not api_key_input:
            st.warning("Se necesita una API Key de Groq para usar el agente.")
        else:
            # Generar resumen dinámico del dataset filtrado
            summary = get_dynamic_eda_summary(df_filtered)
            
            st.markdown("#### Resumen para el Agente:")
            with st.expander("Ver el resumen que recibirá la IA"):
                st.text(summary)

            # Entrada de pregunta del usuario
            user_question = st.text_area(
                "Haz una pregunta específica sobre los jugadores seleccionados:", 
                height=100
            )
            
            # Botón para invocar al agente
            if st.button("Consultar al Agente"):
                if user_question:
                    with st.spinner("El Director Deportivo está analizando los datos..."):
                        response = get_agent_response(api_key_input, summary, user_question)
                        st.success(response)
                else:
                    st.warning("Por favor, introduce una pregunta.")
    # --- TAB 2: Visión General ---
    with tab2:
        st.header("Visión General de los Datos Seleccionados")
        st.dataframe(df_filtered)  # Muestra la tabla de jugadores filtrados

        st.header("Correlación de Métricas")
        st.pyplot(plot_correlation_heatmap(df_filtered))


    # --- TAB 3: Rendimiento ---
    with tab3:
        st.header("Análisis de Rendimiento")
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_top_players(df_filtered, 'Goals', 'Top 10 Goleadores'))
        with col2:
            st.pyplot(plot_top_players(df_filtered, 'Assists', 'Top 10 Asistidores'))

        st.pyplot(plot_top_players(df_filtered, 'Performance', 'Top 10 por Rendimiento Total'))


    # --- TAB 4: Finanzas y Eficiencia ---
    with tab4:
        st.header("Análisis Financiero y de Eficiencia")
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_value_distribution(df_filtered))
        with col2:
            st.pyplot(plot_top_players(df_filtered, 'Market Value', 'Top 10 Jugadores más Valiosos'))

        st.header("Análisis de Eficiencia (Moneyball)")
        st.pyplot(plot_efficiency_scatter(df_filtered))



# --- Caso inicial: no se ha cargado archivo ---
else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis.")
