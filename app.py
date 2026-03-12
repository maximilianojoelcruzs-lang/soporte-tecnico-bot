import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Soporte Técnico Bot | Intelligence Edition",
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

# --- DATA LOADING & EMBEDDINGS ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df['Titulo'] = df['Titulo'].fillna('')
        df['Comentario'] = df['Comentario'].fillna('')
        df['combined_text'] = df['Titulo'] + " " + df['Comentario']
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

def get_embeddings(texts, api_key):
    genai.configure(api_key=api_key)
    # Use embedding-001 for semantic search
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'])

@st.cache_resource
def compute_database_embeddings(df, api_key):
    with st.spinner("Inicializando cerebro semántico (esto solo ocurre una vez)..."):
        # We process in chunks to avoid API limits if the DB is large
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(df), batch_size):
            batch = df['combined_text'].iloc[i:i+batch_size].tolist()
            all_embeddings.extend(get_embeddings(batch, api_key))
        return np.array(all_embeddings)

def get_relevant_context(query, df, db_embeddings, api_key, top_n=12):
    # Get embedding for the user query
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )['embedding']
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], db_embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    context_str = ""
    for i in top_indices:
        row = df.iloc[i]
        context_str += f"- Título: {row['Titulo']}\n  Solución: {row['Comentario']}\n\n"
    return context_str

# --- INITIALIZATION ---
EXCEL_FILE = "Bd  dato.xlsx"
df = load_data(EXCEL_FILE)
api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if api_key and not df.empty:
    db_embeddings = compute_database_embeddings(df, api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Asistente de Soporte Inteligente")
st.markdown("<p style='text-align: center; color: #888;'>Ahora con búsqueda semántica: entiendo sinónimos y conceptos.</p>", unsafe_allow_html=True)

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
            # --- PHASE 1: SEARCH TOP 12 ---
            context = get_relevant_context(prompt, df, db_embeddings, api_key, top_n=12)
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            check_prompt = f"""
Basándote en estos tickets, ¿existe una solución directa o muy relacionada para el problema: "{prompt}"?
Responde solo 'SI' o 'NO'.

Tickets:
{context}
"""
            check_response = model.generate_content(check_prompt).text.strip().upper()
            
            # --- PHASE 2: FALLBACK (DEEP SEARCH) if needed ---
            if "NO" in check_response:
                with st.status("Búsqueda profunda activada... Escaneando toda la base de datos."):
                    # Search TOP 50 for a broader view
                    context = get_relevant_context(prompt, df, db_embeddings, api_key, top_n=50)
                    system_instructions = "No encontré una solución exacta en los primeros resultados, pero analizando toda la base de datos, esto es lo más relevante que encontré:"
            else:
                system_instructions = "He encontrado información relevante en nuestro historial:"

            # final generation
            final_prompt = f"""
{system_instructions}
Eres un experto en soporte. Responde formalmente basado en este contexto:
{context}

Pregunta: {prompt}
"""
            response = model.generate_content(final_prompt, stream=True)
            full_response = ""
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
