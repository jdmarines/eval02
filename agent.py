# agent.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_agent_response(api_key, eda_summary, question):
    """
    Obtiene una respuesta del agente LLM basada en el resumen del EDA y la pregunta del usuario.
    """
    try:
        llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-8b-8192")
        
        prompt_template = """
        Eres un asistente de IA experto en análisis de datos de fútbol, con un enfoque en economía y la filosofía "Moneyball".
        Tu tarea es responder a las preguntas del usuario basándote únicamente en el siguiente resumen del análisis de datos (EDA).
        Sé conciso, directo y profesional en tus respuestas.

        Contexto (Resumen del EDA):
        {eda_summary}

        Pregunta del Usuario:
        {question}

        Respuesta:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({"eda_summary": eda_summary, "question": question})
        return response
    
    except Exception as e:
        return f"Ocurrió un error al contactar al modelo de lenguaje: {e}"
