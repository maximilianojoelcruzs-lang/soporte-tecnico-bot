import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Soporte Técnico AI", page_icon="✨", layout="wide")

# --- PREMIUM CSS STYLING ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@300;400;600&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --accent-color: #a29bfe;
        --bg-dark: #0f111a;
        --card-bg: rgba(255, 255, 255, 0.05);
    }

    /* Global Styles */
    .stApp {
        background-color: var(--bg-dark);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #fff 0%, #a29bfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Glassmorphism Chat Containers */
    .stChatMessage {
        background-color: var(--card-bg) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }

    .stChatMessage:hover {
        transform: translateY(-2px);
        border-color: rgba(162, 155, 254, 0.3) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 17, 26, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Input Field */
    .stChatInputContainer {
        padding-bottom: 2rem !important;
    }

    .stChatInput {
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* Custom Avatar/Icons */
    .stAvatar {
        background: var(--primary-gradient) !important;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage { animation: fadeIn 0.5s ease-out; }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        if 'Titulo' not in df.columns or 'Comentario' not in df.columns:
            st.error("Error: Estructura de Excel no válida.")
            return pd.DataFrame()
        
        df['Titulo'] = df['Titulo'].fillna('')
        df['Comentario'] = df['Comentario'].fillna('')
        df['combined_text'] = df['Titulo'] + " " + df['Comentario']
        return df
    except Exception as e:
        st.error(f"Error cargando base de datos: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_vectorizer_and_matrix(df):
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

def get_top_tickets(query, df, vectorizer, tfidf_matrix, top_n=12):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    top_tickets = df.iloc[top_indices]
    context_str = ""
    for _, row in top_tickets.iterrows():
        context_str += f"- Problema: {row['Titulo']}\n  Solución: {row['Comentario']}\n\n"
        
    return context_str


# --- QUOTA TRACKING LOGIC ---
USAGE_FILE = "api_usage.json"

def load_usage():
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            return json.load(f)
    return {"calls": []}

def save_usage(usage):
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f)

def record_call():
    usage = load_usage()
    now = datetime.now().isoformat()
    usage["calls"].append(now)
    # Mantener solo las últimas 24 horas para no inflar el archivo
    cutoff = (datetime.now() - timedelta(days=1)).isoformat()
    usage["calls"] = [t for t in usage["calls"] if t > cutoff]
    save_usage(usage)

def get_usage_stats():
    usage = load_usage()
    now = datetime.now()
    one_min_ago = (now - timedelta(minutes=1)).isoformat()
    one_day_ago = (now - timedelta(days=1)).isoformat()
    
    rpm = len([t for t in usage["calls"] if t > one_min_ago])
    rpd = len([t for t in usage["calls"] if t > one_day_ago])
    
    return rpm, rpd

# --- INITIALIZATION & SECURITY ---
EXCEL_FILE = "Bd  dato.xlsx"
df = load_data(EXCEL_FILE)

# API Key (Manejo Seguro)
# Se recomienda configurar esto en st.secrets o variables de entorno
INTERNAL_API_KEY = "AIzaSyAmqhqNOX24XSTBhoED-zDdByXkF-NTVH4" 
API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") or INTERNAL_API_KEY

if not df.empty:
    vectorizer, tfidf_matrix = get_vectorizer_and_matrix(df)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI LAYOUT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.markdown("### ✨ Soporte Premium")
    
    # Dashboard de Cuotas
    st.markdown("---")
    st.markdown("#### 📊 Límites de API (Capa Gratis)")
    rpm, rpd = get_usage_stats()
    
    # RPM Progress
    rpm_limit = 15
    rpm_perc = min(rpm / rpm_limit, 1.0)
    st.write(f"Consultas por Minuto: {rpm}/{rpm_limit}")
    st.progress(rpm_perc)
    
    # RPD Progress
    rpd_limit = 1500
    rpd_perc = min(rpd / rpd_limit, 1.0)
    st.write(f"Consultas por Día: {rpd}/{rpd_limit}")
    st.progress(rpd_perc)
    
    if rpm >= rpm_limit:
        st.error("⚠️ Límite por minuto alcanzado. Espera un momento.")
    if rpd >= rpd_limit:
        st.error("🚫 Límite diario agotado.")

    st.markdown("---")
    if st.button("Reiniciar Conversación"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Asistente de Soporte Técnico")
st.markdown("##### Experto en resolución basada en historial técnico.")

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- CHAT INPUT & PROCESSING ---
if prompt := st.chat_input("Describe tu problema técnico..."):
    if not API_KEY:
        st.error("Error crítico: Configuración de API no encontrada.")
        st.stop()
        
    if df.empty:
        st.error("Base de datos no disponible.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Analizando historial y generando solución..."):
                context = get_top_tickets(prompt, df, vectorizer, tfidf_matrix, top_n=12)
            
                genai.configure(api_key=API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
                system_prompt = f"""
Eres un agente de soporte técnico experto de nivel Senior. 
Tu misión es resolver el problema del usuario de forma profesional, clara y estructurada.

DATOS DEL HISTORIAL (TU FUENTE DE VERDAD):
{context}

PROBLEMA ACTUAL:
{prompt}

INSTRUCCIONES DE RESPUESTA:
1. Sé directo y profesional.
2. Si el historial tiene la solución, preséntala paso a paso.
3. Si no hay una coincidencia exacta, usa tu conocimiento técnico para dar una guía lógica basada en el contexto recibido.
4. Mantén un tono cordial y experto en todo momento.
"""
                response = model.generate_content(system_prompt)
                full_response = response.text
                
                # Registrar llamada exitosa
                record_call()
                
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.error("🚨 **Límite de consultas alcanzado.** Has llegado al tope de la capa gratuita de Google AI Studio por el momento. Por favor, espera unos minutos o intenta mañana.")
            else:
                st.error(f"Error en el procesamiento: {e}")

