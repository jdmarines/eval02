# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Feature Engineering ---
def create_features(df):
    """Crea nuevas columnas para un análisis más profundo."""
    df_copy = df.copy()
    df_copy['Performance'] = df_copy['Goals'] + df_copy['Assists']
    
    # Calcular eficiencia (costo por contribución a gol)
    df_copy['Cost_per_Performance'] = df_copy.apply(
        lambda row: row['Market Value'] / row['Performance'] if row['Performance'] > 0 else 0,
        axis=1
    )
    
    # Categorías de edad
    bins = [0, 21, 29, 40]
    labels = ['Joven Promesa (<=21)', 'En su Prime (22-29)', 'Veterano (30+)']
    df_copy['Age Group'] = pd.cut(df_copy['Age'], bins=bins, labels=labels, right=True)
    
    return df_copy

# --- Funciones de Visualización ---
def plot_correlation_heatmap(df):
    """Muestra la correlación entre las variables numéricas."""
    fig, ax = plt.subplots(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Mapa de Calor de Correlación de Variables Numéricas')
    return fig

def plot_value_distribution(df):
    """Muestra la distribución del valor de mercado."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Market Value'] / 1_000_000, kde=True, ax=ax, bins=20)
    ax.set_title('Distribución del Valor de Mercado (en Millones de EUR)')
    ax.set_xlabel('Valor de Mercado (Millones de EUR)')
    ax.set_ylabel('Número de Jugadores')
    return fig

def plot_top_players(df, metric, title):
    """Muestra un gráfico de barras de los 10 mejores jugadores por una métrica."""
    top_10 = df.nlargest(10, metric)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_10, x=metric, y='Name', palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel('Jugador')
    return fig

def plot_efficiency_scatter(df):
    """Gráfico de dispersión para analizar la eficiencia (Moneyball)."""
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
    ax.set_yscale('log')
    ax.set_title('Análisis de Eficiencia: Valor vs. Rendimiento')
    ax.set_xlabel('Rendimiento Total (Goles + Asistencias)')
    ax.set_ylabel('Valor de Mercado (EUR) - Escala Logarítmica')
    ax.legend(title='Grupo de Edad')
    return fig

# --- Función para el Resumen del Agente ---
def get_dynamic_eda_summary(df):
    """
    Genera un informe de scouting detallado y estructurado para el agente de IA.
    """
    if df.empty:
        return "No hay jugadores que coincidan con los filtros seleccionados."

    summary = f"**INFORME DE SCOUTING PARA {df.shape[0]} JUGADORES**\n\n"
    summary += f"**Visión General:**\n"
    summary += f"- **Edad Promedio:** {df['Age'].mean():.1f} años\n"
    summary += f"- **Valor de Mercado Promedio:** {df['Market Value'].mean():,.0f} EUR\n"
    if not df['Club'].empty:
        summary += f"- **Club con más Jugadores:** {df['Club'].mode().iloc[0]}\n"
    summary += "--- \n"

    summary += "**Análisis por Posición:**\n"
    
    forwards = df[df['Position'].str.contains('Forward|Winger', na=False)]
    if not forwards.empty:
        avg_perf_fwd = forwards['Performance'].mean()
        top_forward = forwards.loc[forwards['Performance'].idxmax()]
        summary += f"- **Mejor Delantero:** {top_forward['Name']} ({top_forward['Club']}) con {top_forward['Performance']} contribuciones (Promedio: {avg_perf_fwd:.1f}).\n"

    midfielders = df[df['Position'].str.contains('Midfield', na=False)]
    if not midfielders.empty:
        avg_perf_mid = midfielders['Performance'].mean()
        top_midfielder = midfielders.loc[midfielders['Performance'].idxmax()]
        summary += f"- **Mejor Mediocampista:** {top_midfielder['Name']} ({top_midfielder['Club']}) con {top_midfielder['Performance']} contribuciones (Promedio: {avg_perf_mid:.1f}).\n"

    defenders = df[df['Position'].str.contains('Centre-Back|Full-Back', na=False)]
    if not defenders.empty:
        top_defender = defenders.loc[defenders['Market Value'].idxmax()]
        summary += f"- **Mejor Defensor (por Valor):** {top_defender['Name']} ({top_defender['Club']}) valorado en {top_defender['Market Value']:,.0f} EUR.\n"
    summary += "---\n"

    summary += "**Hallazgos Clave:**\n"

    performers = df[df['Cost_per_Performance'] > 0]
    if not performers.empty:
        most_efficient = performers.loc[performers['Cost_per_Performance'].idxmin()]
        summary += f"- **Jugador más Eficiente (Moneyball):** {most_efficient['Name']} ({most_efficient['Club']}), costo de {most_efficient['Cost_per_Performance']:,.0f} EUR por contribución.\n"
    
    young_players = df[df['Age'] <= 21]
    if not young_players.empty:
        top_young_performer = young_players.loc[young_players['Performance'].idxmax()]
        summary += f"- **Joven Promesa Destacada:** {top_young_performer['Name']} ({top_young_performer['Age']} años), el sub-21 con mejor rendimiento ({top_young_performer['Performance']} contribuciones).\n"
        
    veteran_players = df[df['Age'] >= 30]
    if not veteran_players.empty:
        top_veteran_performer = veteran_players.loc[veteran_players['Performance'].idxmax()]
        summary += f"- **Veterano de Impacto Inmediato:** {top_veteran_performer['Name']} ({top_veteran_performer['Age']} años), el mayor de 30 con mejor rendimiento ({top_veteran_performer['Performance']} contribuciones).\n"
        
    df['Total Cards'] = df['Yellow Cards'] + df['Red Cards']
    if not df.empty and df['Total Cards'].max() > 0:
        riskiest_player = df.loc[df['Total Cards'].idxmax()]
        summary += f"- **Riesgo Disciplinario:** {riskiest_player['Name']} es el jugador con más tarjetas ({riskiest_player['Total Cards']}).\n"
        
    return summary
