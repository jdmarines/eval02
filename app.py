# app.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Importar funciones de nuestros módulos
from eda import get_eda_summary, plot_age_distribution, plot_moneyball_analysis
from agent import get_agent_response

# Cargar la API key desde el archivo .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Panel Moneyball")
st.title("📊 Panel de Análisis 'Moneyball' para Fichajes")

# --- Sidebar ---
st.sidebar.header("Configuración")
api_key_input = st.sidebar.text_input(
    "Introduce tu API Key de Groq", 
    type="password", 
    value=GROQ_API_KEY or "",
    help="Puedes obtener tu clave en https://console.groq.com/keys"
)

st.sidebar.markdown("---")
st.sidebar.info("Esta aplicación te permite analizar datos de jugadores y obtener recomendaciones de una IA.")

# --- Lógica Principal de la Aplicación ---
uploaded_file = st.file_uploader("Sube tu archivo CSV con estadísticas de jugadores", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Archivo CSV cargado exitosamente!")
        st.write("### Vista Previa de los Datos:")
        st.dataframe(df.head())

        # --- Sección de Análisis Exploratorio de Datos (EDA) ---
        st.markdown("---")
        st.header("🔎 Análisis Exploratorio de Datos (EDA)")
        
        eda_summary = get_eda_summary(df)
        st.write("#### Resumen del Análisis:")
        st.text(eda_summary)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Distribución de Edades")
            fig_age = plot_age_distribution(df)
            st.pyplot(fig_age)
        
        with col2:
            st.write("#### Gráfico de Análisis 'Moneyball'")
            fig_moneyball = plot_moneyball_analysis(df)
            if fig_moneyball:
                st.pyplot(fig_moneyball)
            else:
                st.warning("El archivo CSV debe contener las columnas 'Goals' y 'Assists' para el gráfico Moneyball.")

        # --- Sección del Agente Inteligente ---
        st.markdown("---")
        st.header("🤖 Asistente de Fichajes (IA)")

        if not api_key_input:
            st.warning("Por favor, introduce tu API Key de Groq en la barra lateral para activar el asistente.")
        else:
            user_question = st.text_area("Haz una pregunta sobre los datos analizados:", height=100)
            
            if st.button("Obtener Recomendación"):
                if user_question:
                    with st.spinner("El asistente está pensando..."):
                        response = get_agent_response(api_key_input, eda_summary, user_question)
                        st.info(response)
                else:
                    st.warning("Por favor, escribe una pregunta.")
                    
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("Esperando que subas un archivo CSV para comenzar el análisis.")
