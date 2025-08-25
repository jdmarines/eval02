# ⚽ Interactive Football Scouting Dashboard

Un dashboard interactivo para scouting de fútbol construido con **Streamlit**, diseñado para que analistas, ojeadores y aficionados puedan explorar datos de jugadores.
Permite subir tu propio archivo CSV, aplicar filtros dinámicos y obtener insights sobre métricas de rendimiento y financieras.
Además, incluye un **agente de IA** que responde preguntas en lenguaje natural sobre el dataset filtrado en ese momento.

---

## ✨ Features

🔹 **Interactive Dashboard**

* Interfaz limpia e intuitiva, desarrollada con Streamlit.

🔹 **Dynamic Filtering**

* Filtra jugadores por:

  * Club
  * Nacionalidad principal
  * Posición
  * Rango de edad

🔹 **Performance Analysis**

* Visualiza el Top 10 de jugadores por:

  * Goles
  * Asistencias
  * Puntaje combinado de rendimiento (Goles + Asistencias)

🔹 **Financial & Efficiency Analysis**

* Distribución de valores de mercado.
* Top 10 jugadores más valiosos.
* Gráfico estilo *Moneyball*:

  * **Eje X:** Valor de mercado
  * **Eje Y:** Puntaje de rendimiento
  * Permite identificar talento infravalorado.

🔹 **AI Scouting Agent**

* Basado en **LangChain + Groq (Llama3)**.
* Entiende preguntas en lenguaje natural.
* Las respuestas siempre están fundamentadas en el dataset filtrado (evitando alucinaciones).

---

## ⚙️ Technical Details

### 🧩 `eda.py` → Exploración de Datos y Feature Engineering

Este módulo se encarga del preprocesamiento y generación de nuevas variables:

* **Performance** = Goles + Asistencias.
* **Cost per Performance** = Valor de mercado / Performance → menor valor = jugador más costo-eficiente.
* **Agrupación por edad**:

  * `Joven Promesa` (≤21)
  * `En su Prime` (22–29)
  * `Veterano` (30+)
* Generación de gráficos y tablas interactivas para visualización en el dashboard.

💡 *Comentario*: Este script asegura que los datos estén listos para análisis antes de mostrarlos en la interfaz.

---

### 🤖 `agent.py` → Agente de Inteligencia Artificial

Aquí vive la lógica del **AI Scouting Agent**:

* Implementación de **LangChain** con el modelo **Groq (Llama3)**.
* Función `get_dynamic_eda_summary`: genera un resumen dinámico de los datos filtrados.
* Se envían tanto la consulta del usuario como el resumen al modelo LLM.
* El *prompt* está diseñado para garantizar que las respuestas sean:

  * Contextuales
  * Relevantes al dataset cargado
  * Sin información inventada

💡 *Comentario*: Este archivo es el “cerebro” del asistente que interpreta lenguaje natural.

---

### 🎨 `app.py` → Entrada Principal de la Aplicación

Archivo central que lanza la aplicación con **Streamlit**.
Funciones principales:

1. Configura la interfaz del dashboard.
2. Carga el dataset inicial (o permite al usuario subir uno nuevo).
3. Aplica filtros dinámicos (club, nacionalidad, edad, posición).
4. Renderiza gráficas de rendimiento y finanzas usando `eda.py`.
5. Conecta con el agente de IA de `agent.py` para responder consultas.

💡 *Comentario*: Es el **núcleo integrador** que combina visualización + análisis + agente de IA.

---

### 📦 `requirements.txt`

Lista de dependencias necesarias para ejecutar el proyecto:

* `streamlit`
* `pandas`
* `matplotlib` / `plotly` (para gráficas)
* `langchain`
* `groq`
* `python-dotenv` (manejo de variables de entorno)

💡 *Comentario*: Asegura la reproducibilidad del entorno de ejecución.

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.8+
* pip

### 🛠️ Generador de CSV

El proyecto cuenta con un módulo adicional para generar datasets personalizados de jugadores en formato .csv, ideal para pruebas y análisis.

Este generador se encuentra en el repositorio:
👉 Player.csv_generator

Uso:

Clonar el repositorio:

git clone https://github.com/jmpenam1/Player.csv_generator
cd Player.csv_generator


Ejecutar el generador:

Generator.py

Se abrirá un formulario en el navegador donde podrás ingresar los datos de jugadores.
Al finalizar, se guardará un archivo players.csv compatible con este proyecto principal.

URL: https://playercsvgenerator-j8mnbbn5cprychc3c2f8dm.streamlit.app/

### 📥 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/jdmarines/eval02.git
cd eval02

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
# Crear archivo .env en la raíz del proyecto
GROQ_API_KEY="tu_api_key_aqui"

# 4. Ejecutar la app
streamlit run app.py
```

Abre la URL local mostrada en la terminal.

---

## 📂 Project Structure

```
.
├── README.md                  # Documentación del proyecto
├── agent.py                   # Lógica del agente de IA (LangChain + Groq)
├── app.py                     # Punto de entrada del dashboard en Streamlit
├── eda.py                     # Preprocesamiento, ingeniería de variables y visualización
├── requirements.txt           # Dependencias necesarias
└── dataset/
    └── Top 500 Players 2024.csv   # Dataset de prueba incluido
```

---

## 📊 Demo Dataset

* Se incluye un CSV de ejemplo (`Top 500 Players 2024.csv`) en la carpeta `/dataset`.
* También puedes cargar tu propio dataset para personalizar el análisis.

---

## 💡 Future Improvements

* Incorporar métricas avanzadas (xG, pases clave, duelos ganados).
* Comparación directa entre jugadores.
* Exportación de datos filtrados + reportes generados por el agente.

---
