import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Soporte Técnico Bot | Fast Edition",
    page_icon="🤖",
    layout="centered"
)

# --- PREMIUM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: rgba(26, 28, 35, 0.8); backdrop-filter: blur(10px); }
    h1 {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df['Titulo'] = df['Titulo'].fillna('')
        df['Comentario'] = df['Comentario'].fillna('')
        df['combined_text'] = df['Titulo'] + " " + df['Comentario']
        return df
    except Exception as e:
        st.error(f"Error cargando Excel: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_search_matrix(df):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, matrix

def get_top_context(query, df, vectorizer, matrix, top_n=20):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    context = ""
    for i in top_indices:
        row = df.iloc[i]
        context += f"- Título: {row['Titulo']}\n  Solución: {row['Comentario']}\n\n"
    return context

# --- INITIALIZATION ---
EXCEL_FILE = "Bd  dato.xlsx"
df = load_data(EXCEL_FILE)

# API Key: Internal fallback
INTERNAL_API_KEY = "AIzaSyAmqhqNOX24XSTBhoED-zDdByXkF-NTVH4"
api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or INTERNAL_API_KEY

if not df.empty:
    vectorizer, matrix = get_search_matrix(df)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI ---
st.title("🤖 Asistente de Soporte Técnico")
st.markdown("<p style='text-align: center; color: #888;'>Optimizado para respuestas rápidas basadas en el historial.</p>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    if not api_key:
        st.warning("Configura tu API Key.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 1. Get Top 20 Context (Fast TF-IDF)
            context = get_top_context(prompt, df, vectorizer, matrix, top_n=20)
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-flash-latest')
            
            # 2. Direct Generation
            system_prompt = f"""
Eres un experto en soporte técnico. Responde basándote en este contexto del historial de tickets:
{context}

Si el problema no tiene una solución clara en el contexto, sugiere pasos generales técnicos de forma profesional.

Pregunta del usuario: {prompt}
"""
            response = model.generate_content(system_prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error generando respuesta: {e}")
