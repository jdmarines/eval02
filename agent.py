# agent.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_agent_response(api_key, eda_summary, question):
    """
    Obtiene una respuesta del agente LLM basada en el resumen del EDA y la pregunta del usuario.
    """
    try:
        llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")
        
        # --- PROMPT MEJORADO ---
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
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({"eda_summary": eda_summary, "question": question})
        return response
    
    except Exception as e:
        return f"Ocurrió un error al contactar al modelo de lenguaje: {e}"
