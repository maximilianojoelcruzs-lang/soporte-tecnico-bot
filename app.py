import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Soporte Técnico Bot", page_icon="🤖", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    .chat-container {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .user-msg {
        background-color: #2b313e;
        color: white;
    }
    .bot-msg {
        background-color: #1a1c23;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data(file_path):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        # Ensure 'Titulo' and 'Comentario' exist, filling NaNs
        if 'Titulo' not in df.columns or 'Comentario' not in df.columns:
            st.error("El archivo Excel debe contener las columnas 'Titulo' y 'Comentario'.")
            return pd.DataFrame()
        
        df['Titulo'] = df['Titulo'].fillna('')
        df['Comentario'] = df['Comentario'].fillna('')
        
        # Create a combined text for vectorization
        df['combined_text'] = df['Titulo'] + " " + df['Comentario']
        return df
    except Exception as e:
        st.error(f"Error cargando el archivo Excel: {e}")
        return pd.DataFrame()

# Initialize Vectorizer for fast filtering
@st.cache_resource
def get_vectorizer_and_matrix(df):
    vectorizer = TfidfVectorizer(stop_words=None) # We can add spanish stop words if needed
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

# Function to get top K relevant tickets
def get_top_tickets(query, df, vectorizer, tfidf_matrix, top_n=12):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Filter the exact top tickets
    top_tickets = df.iloc[top_indices]
    
    # Create a string representation for the LLM
    context_str = ""
    for i, row in top_tickets.iterrows():
        context_str += f"- Problema (Título): {row['Titulo']}\n"
        context_str += f"  Solución/Comentario: {row['Comentario']}\n\n"
        
    return context_str


# --- INITIALIZATION ---
EXCEL_FILE = "Bd  dato.xlsx"
df = load_data(EXCEL_FILE)

if not df.empty:
    vectorizer, tfidf_matrix = get_vectorizer_and_matrix(df)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ask for Gemini API Key if not set in environment
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")


st.title("🤖 Asistente de Soporte Técnico")
st.markdown("Soy tu agente de soporte. Describe el error o problema que tienes y buscaré la mejor solución basada en nuestro historial de tickets.")


# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- CHAT INPUT & PROCESSING ---
if prompt := st.chat_input("Escribe tu problema aquí (ej. 'Error al conectar base de datos')"):
    if not api_key:
        st.warning("Por favor, ingresa tu API Key de Gemini en la barra lateral para continuar.")
        st.stop()
        
    if df.empty:
        st.error("No se pudo cargar la base de datos de conocimiento.")
        st.stop()
        
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 1. Retrieve top 12 tickets based on user prompt
            with st.spinner("Buscando soluciones previas..."):
                context = get_top_tickets(prompt, df, vectorizer, tfidf_matrix, top_n=12)
            
            # 2. Configure Gemini GenAI
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # 3. Build the prompt
            system_prompt = f"""
Eres un agente de soporte técnico experto, formal y lógico. 
Tu objetivo es ayudar al usuario a resolver su problema de software basándote EXCLUSIVAMENTE en el historial de tickets previos que te proporcionaré como contexto.

Instrucciones:
1. Analiza el problema del usuario.
2. Revisa el historial de tickets para ver si hay un error similar y cómo se solucionó (fíjate en la sección 'Solución/Comentario' que son las respuestas que se han dado).
3. Si encuentras una solución relevante, explícasela al usuario de manera clara, estructurada y formal, como si fueras un profesional de soporte.
4. Puedes decir algo como "Este error se solucionó de esta manera:" y dar los pasos o la lógica.
5. Si no encuentras nada relevante en el contexto proporcionado, pide más detalles cordialmente o dile que este error específico no parece estar documentado.

Contexto (Historial de los 12 tickets más relevantes):
{context}

Problema del Usuario:
{prompt}
"""
            # 4. Get response from Gemini
            with st.spinner("Generando respuesta..."):
                response = model.generate_content(system_prompt)
                full_response = response.text
                
            # Display response
            message_placeholder.markdown(full_response)
            
            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Ocurrió un error al procesar la solicitud: {e}")
