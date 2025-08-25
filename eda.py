# eda.py
# --------------------------------------------------
# Este módulo contiene las funciones de:
# 1. Feature Engineering → creación de nuevas variables a partir del dataset.
# 2. Visualización → gráficos para explorar rendimiento, finanzas y eficiencia.
# 3. Resumen dinámico → generar un texto descriptivo del dataset filtrado
#    que servirá como entrada al agente de IA.
# --------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- FEATURE ENGINEERING ---
def create_features(df):
    """
    Crea nuevas columnas para un análisis más profundo.
    
    - Performance: suma de goles y asistencias.
    - Cost_per_Performance: coste por contribución ofensiva (valor / performance).
    - Age Group: clasificación de los jugadores en categorías de edad.
    
    Retorna un DataFrame con las columnas adicionales.
    """
    df_copy = df.copy()
    df_copy['Performance'] = df_copy['Goals'] + df_copy['Assists']
    
    # Eficiencia (evita dividir por cero si Performance = 0)
    df_copy['Cost_per_Performance'] = df_copy.apply(
        lambda row: row['Market Value'] / row['Performance'] if row['Performance'] > 0 else 0,
        axis=1
    )
    
    # Categorizar edades en tres grupos
    bins = [0, 21, 29, 40]
    labels = ['Joven Promesa (<=21)', 'En su Prime (22-29)', 'Veterano (30+)']
    df_copy['Age Group'] = pd.cut(df_copy['Age'], bins=bins, labels=labels, right=True)
    
    return df_copy


# --- FUNCIONES DE VISUALIZACIÓN ---
def plot_correlation_heatmap(df):
    """
    Genera un mapa de calor de correlación entre variables numéricas.
    Útil para identificar relaciones entre métricas.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Mapa de Calor de Correlación de Variables Numéricas')
    return fig


def plot_value_distribution(df):
    """
    Muestra la distribución del valor de mercado de los jugadores.
    Escalado en millones de euros para facilitar interpretación.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Market Value'] / 1_000_000, kde=True, ax=ax, bins=20)
    ax.set_title('Distribución del Valor de Mercado (en Millones de EUR)')
    ax.set_xlabel('Valor de Mercado (Millones de EUR)')
    ax.set_ylabel('Número de Jugadores')
    return fig


def plot_top_players(df, metric, title):
    """
    Muestra los 10 mejores jugadores según una métrica específica.
    Ejemplos: Goals, Assists, Performance, Market Value.
    """
    top_10 = df.nlargest(10, metric)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_10, x=metric, y='Name', palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel('Jugador')
    return fig


def plot_efficiency_scatter(df):
    """
    Scatter plot estilo "Moneyball":
    - Eje X: Rendimiento total (Goles + Asistencias).
    - Eje Y: Valor de mercado (escala logarítmica).
    - Color: Grupo de edad.
    - Tamaño: Coste por contribución.
    
    Permite identificar talento infravalorado.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df[df['Performance'] > 0],
        x='Performance',
        y='Market Value',
        hue='Age Group',
        size='Cost_per_Performance',
        sizes=(50, 1000),
        alpha=0.7,
        palette='magma',
        ax=ax
    )
    ax.set_yscale('log')  # Escala log → más legible al haber gran dispersión de valores
    ax.set_title('Análisis de Eficiencia: Valor vs. Rendimiento')
    ax.set_xlabel('Rendimiento Total (Goles + Asistencias)')
    ax.set_ylabel('Valor de Mercado (EUR) - Escala Logarítmica')
    ax.legend(title='Grupo de Edad')
    return fig


# --- RESUMEN DINÁMICO PARA EL AGENTE ---
def get_dynamic_eda_summary(df):
    """
    Genera un resumen textual de los datos filtrados.
    Incluye:
    - Cantidad de jugadores seleccionados.
    - Edad y valor de mercado promedio.
    - Club más frecuente.
    - Top 5 jugadores por rendimiento.
    - Top 5 jugadores más eficientes (mejor relación costo/rendimiento).
    
    Este resumen se utiliza como "fuente de verdad"
    para que el agente de IA responda preguntas.
    """
    if df.empty:
        return "No hay jugadores que coincidan con los filtros seleccionados."

    summary = f"Resumen del análisis para los {df.shape[0]} jugadores seleccionados:\n\n"
    
    summary += f"**Visión General:**\n"
    summary += f"- Edad promedio: {df['Age'].mean():.1f} años.\n"
    summary += f"- Valor de mercado promedio: {df['Market Value'].mean():,.0f} EUR.\n"
    summary += f"- Club más representado: {df['Club'].mode().iloc[0]}.\n"
    
    # Top 5 por rendimiento
    df_perf = df.nlargest(5, 'Performance')
    summary += "\n**Top 5 Jugadores por Rendimiento (Goles + Asistencias):**\n"
    for _, row in df_perf.iterrows():
        summary += f"- {row['Name']} ({row['Club']}): {row['Performance']} contribuciones.\n"
        
    # Top 5 por eficiencia (si hay datos válidos)
    performers = df[df['Cost_per_Performance'] > 0]
    if not performers.empty:
        best_value_players = performers.nsmallest(5, 'Cost_per_Performance')
        summary += "\n**Top 5 Jugadores más Eficientes (Menor Costo por Rendimiento):**\n"
        for _, row in best_value_players.iterrows():
            summary += f"- {row['Name']} ({row['Club']}): {row['Cost_per_Performance']:,.0f} EUR por contribución.\n"
            
    return summary
