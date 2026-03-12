import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

# --- SEARCH LOGIC (EMBEDDINGS & TF-IDF) ---

def try_get_embedding(text, model_name, task_type):
    try:
        return genai.embed_content(
            model=model_name,
            content=text,
            task_type=task_type
        )['embedding']
    except:
        return None

def get_embeddings_resilient(texts, api_key):
    genai.configure(api_key=api_key)
    # Use the model name discovered via diagnostics
    model_name = "models/gemini-embedding-001"
    try:
        emb = try_get_embedding(texts, model_name, "retrieval_document")
        if emb is not None:
            return np.array(emb), model_name
    except:
        pass
    return None, None

@st.cache_resource
def prepare_search_engines(df, api_key):
    """Initializes both Embeddings and TF-IDF as fallback"""
    engines = {"type": "tfidf", "vectorizer": None, "matrix": None, "embeddings": None, "model": None}
    
    # 1. Try to initialize Embeddings
    if api_key:
        with st.spinner("Inicializando cerebro semántico..."):
            try:
                # We only check the first row to see if the model works
                test_emb, active_model = get_embeddings_resilient([df['combined_text'].iloc[0]], api_key)
                if active_model:
                    # Compute all embeddings
                    all_embs = []
                    batch_size = 50
                    for i in range(0, len(df), batch_size):
                        batch = df['combined_text'].iloc[i:i+batch_size].tolist()
                        batch_res, _ = get_embeddings_resilient(batch, api_key)
                        all_embs.extend(batch_res)
                    engines["type"] = "embeddings"
                    engines["embeddings"] = np.array(all_embs)
                    engines["model"] = active_model
            except:
                pass

    # 2. Always prepare TF-IDF as fallback
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df['combined_text'])
    engines["vectorizer"] = vectorizer
    engines["matrix"] = matrix
    
    return engines

def get_relevant_context_resilient(query, df, engines, api_key, top_n=12):
    if engines["type"] == "embeddings" and api_key:
        try:
            # Use the same model as initialization
            query_emb = try_get_embedding(query, engines["model"], "retrieval_query")
            if query_emb:
                similarities = cosine_similarity([query_emb], engines["embeddings"]).flatten()
                top_indices = similarities.argsort()[-top_n:][::-1]
                context = ""
                for i in top_indices:
                    row = df.iloc[i]
                    context += f"- Título: {row['Titulo']}\n  Solución: {row['Comentario']}\n\n"
                return context, True
        except:
            pass
    
    # Fallback to TF-IDF
    query_vec = engines["vectorizer"].transform([query])
    similarities = cosine_similarity(query_vec, engines["matrix"]).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    context = ""
    for i in top_indices:
        row = df.iloc[i]
        context += f"- Título: {row['Titulo']}\n  Solución: {row['Comentario']}\n\n"
    return context, False

# --- INITIALIZATION ---
EXCEL_FILE = "Bd  dato.xlsx"
df = load_data(EXCEL_FILE)

# API Key: Prioritize Secrets/Env, then use the provided Internal Key
INTERNAL_API_KEY = "AIzaSyAmqhqNOX24XSTBhoED-zDdByXkF-NTVH4"
api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or INTERNAL_API_KEY

engines = None
if not df.empty and api_key:
    engines = prepare_search_engines(df, api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI ---
st.title("🤖 Asistente de Soporte Inteligente")
mode_label = "🧠 Modo Semántico" if engines and engines["type"] == "embeddings" else "🔍 Modo Estándar (Fallback)"
st.markdown(f"<p style='text-align: center; color: #888;'>{mode_label} activado.</p>", unsafe_allow_html=True)

# Warning if no key is found at all (shouldn't happen with INTERNAL_API_KEY)
if not api_key:
    st.sidebar.warning("⚠️ No se detectó API Key. Ingrésala manualmente:")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    if not api_key:
        st.warning("Configura tu API Key en la barra lateral.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 1. Search Context
            context, is_semantic = get_relevant_context_resilient(prompt, df, engines, api_key, top_n=12)
            
            genai.configure(api_key=api_key)
            # Use 'gemini-flash-latest' which is available for this API Key
            model = genai.GenerativeModel('gemini-flash-latest')
            
            # 2. Check Relevance and Generate Response
            check_prompt = f"Basándote en estos tickets, ¿existe una solución para: '{prompt}'? Responde solo SI o NO.\n\nContexto:\n{context}"
            check_response = model.generate_content(check_prompt).text.strip().upper()
            
            # Deep search if No
            if "NO" in check_response:
                with st.status("Búsqueda profunda activada..."):
                    context, _ = get_relevant_context_resilient(prompt, df, engines, api_key, top_n=50)
                system_instr = "Analizando todo el historial disponible para encontrar una pista..."
            else:
                system_instr = "He encontrado esta información relevante en el historial:"

            final_prompt = f"{system_instr}\n\nContexto:\n{context}\n\nUsuario: {prompt}"
            response = model.generate_content(final_prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error generando respuesta: {e}")
