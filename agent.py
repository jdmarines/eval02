# agent.py
# --------------------------------------------------
# Este módulo contiene la lógica del "Agente de Scouting".
# Se encarga de conectarse al modelo de lenguaje (LLM) vía la API de Groq, construir un prompt con las 
# instrucciones específicas, y devolver una respuesta basada ÚNICAMENTE en el resumen dinámico del dataset.
# --------------------------------------------------

# --- Importaciones necesarias ---
from langchain_groq import ChatGroq                       # Cliente para conectarse al LLM vía Groq
from langchain_core.prompts import ChatPromptTemplate     # Para estructurar el prompt con variables dinámicas
from langchain_core.output_parsers import StrOutputParser # Para procesar la respuesta del modelo como texto plano


# --- Función principal: get_agent_response ---
def get_agent_response(api_key, eda_summary, question):
    """
    Obtiene una respuesta del agente LLM basada en el resumen del EDA y la pregunta del usuario.
    
    Parámetros:
    - api_key (str): clave de la API de Groq.
    - eda_summary (str): resumen dinámico de los datos filtrados (EDA).
    - question (str): pregunta formulada por el usuario en lenguaje natural.
    
    Retorna:
    - (str): respuesta generada por el modelo, o un mensaje de error en caso de fallo.
    """
    try:
        # 1. Inicializar el modelo LLM
        # Se usa Llama 3 (70B) con temperatura 0 → respuestas determinísticas.
        llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama3-70b-8192"
        )
        
        # 2. Definir el prompt
        # El prompt fuerza al modelo a actuar como analista de fútbol
        # y a usar ÚNICAMENTE la información en eda_summary.
        prompt_template = """
        Eres un analista de datos de fútbol profesional. Tu única fuente de verdad es el siguiente "Resumen de Datos".
        Debes responder la "Pregunta del Usuario" basándote exclusivamente en la información contenida en el "Resumen de Datos".
        No uses ningún conocimiento externo. Si la pregunta no se puede responder con el resumen, indica que no tienes suficiente información.
        Si la respuesta incluye un jugador, menciona su nombre tal como aparece en el resumen.

        ---
        **Resumen de Datos:**
        {eda_summary}
        ---
        **Pregunta del Usuario:**
        {question}
        ---
        **Respuesta del Analista:**
        """
        
        # 3. Construcción de la cadena (Pipeline)
        # Prompt dinámico → Modelo LLM → Procesador de salida en texto
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        # 4. Invocación de la cadena con los valores reales
        response = chain.invoke({
            "eda_summary": eda_summary,
            "question": question
        })
        
        # 5. Devolver la respuesta del agente
        return response
    
    except Exception as e:
        # Manejo de errores (ej. problemas con la API)
        return f"Ocurrió un error al contactar al modelo de lenguaje: {e}"
