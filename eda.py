# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_eda_summary(df):
    """Genera un resumen textual del EDA para alimentar al LLM."""
    summary = "Resumen del Análisis Exploratorio de Datos (EDA):\n"
    summary += f"El dataset contiene {df.shape[0]} jugadores y {df.shape[1]} columnas.\n"
    null_values = df.isnull().sum().sum()
    summary += f"Se encontraron {null_values} valores nulos en total.\n"
    avg_age = df['Age'].mean()
    avg_value = df['Market Value'].mean()
    summary += f"La edad promedio de los jugadores es {avg_age:.1f} años.\n"
    summary += f"El valor de mercado promedio es {avg_value:,.0f} euros.\n"
    
    if 'Goals' in df.columns and 'Assists' in df.columns:
        df['Performance'] = df['Goals'] + df['Assists']
        # Evitar division por cero si Performance es 0
        df['Value_per_Performance'] = df.apply(
            lambda row: row['Market Value'] / row['Performance'] if row['Performance'] > 0 else 0,
            axis=1
        )
        
        # Filtrar jugadores con rendimiento para encontrar el más infravalorado
        performers = df[df['Performance'] > 0]
        if not performers.empty:
            undervalued_Name = performers.loc[performers['Value_per_Performance'].idxmin()]
            summary += f"El jugador potencialmente más infravalorado (menor costo por contribución a gol) es {undervalued_Name['Name']} "
            summary += f"con un costo de {undervalued_Name['Value_per_Performance']:,.0f} euros por gol o asistencia.\n"
    
    return summary

def plot_age_distribution(df):
    """Crea un histograma de la distribución de edades."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribución de Edades de los Jugadores')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Frecuencia')
    ax.grid(True)
    return fig

def plot_moneyball_analysis(df):
    """Crea un gráfico de dispersión para el análisis Moneyball."""
    if 'Goals' in df.columns and 'Assists' in df.columns:
        # Asegurarse de que la columna Performance existe
        if 'Performance' not in df.columns:
             df['Performance'] = df['Goals'] + df['Assists']
             
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x='Performance', y='Market Value', hue='Position', size='Age', sizes=(50, 500), alpha=0.7, ax=ax)
        ax.set_title('Análisis Moneyball: Valor de Mercado vs. Rendimiento (Goles + Asistencias)')
        ax.set_xlabel('Rendimiento (Goles + Asistencias)')
        ax.set_ylabel('Valor de Mercado (en Euros)')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend(title='Posición', bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig
    return None
