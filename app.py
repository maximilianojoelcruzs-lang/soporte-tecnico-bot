import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Soporte Técnico Bot | Premium Edition",
    page_icon="🤖",
    layout="centered" # Centered for a more focused feel
)

# --- PREMIUM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(26, 28, 35, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat Message Bubbles */
    .chat-message-container {
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.2s ease;
    }
    
    .chat-message-container:hover {
        transform: translateY(-2px);
    }
    
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Input Field Styling */
    .stChatInputContainer {
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        if 'Titulo' not in df.columns or 'Comentario' not in df.columns:
            st.error("El archivo Excel debe contener las columnas 'Titulo' y 'Comentario'.")
            return pd.DataFrame()
        
        df['Titulo'] = df['Titulo'].fillna('')
        df['Comentario'] = df['Comentario'].fillna('')
        df['combined_text'] = df['Titulo'] + " " + df['Comentario']
        return df
    except Exception as e:
        st.error(f"Error cargando el archivo Excel: {e}")
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
    for i, row in top_tickets.iterrows():
        context_str += f"- Problema (Título): {row['Titulo']}\n"
        context_str += f"  Solución/Comentario: {row['Comentario']}\n\n"
        
    return context_str


# --- INITIALIZATION ---
EXCEL_FILE = "Bd  dato.xlsx"
df = load_data(EXCEL_FILE)

if not df.empty:
    vectorizer, tfidf_matrix = get_vectorizer_and_matrix(df)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API KEY HANDLING (Secrets or Sidebar) ---
api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")
    if not api_key:
        st.sidebar.info("Para que la app funcione sin pedir la clave, configúrala en el panel de Streamlit Secrets.")

st.title("🤖 Asistente de Soporte Técnico")
st.markdown("<p style='text-align: center; color: #888;'>Desarrollado para ofrecer soluciones rápidas basadas en el historial técnico.</p>", unsafe_allow_html=True)


# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- CHAT INPUT & PROCESSING ---
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    if not api_key:
        st.warning("Por favor, ingresa tu API Key de Gemini en la barra lateral para continuar.")
        st.stop()
        
    if df.empty:
        st.error("No se pudo cargar la base de datos de conocimiento.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 1. Retrieve top 12 tickets
            context = get_top_tickets(prompt, df, vectorizer, tfidf_matrix, top_n=12)
            
            # 2. Configure Gemini GenAI
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash') # Using stable 1.5 flash
            
            # 3. Build the prompt
            system_prompt = f"""
Eres un agente de soporte técnico experto, formal y lógico. 
Tu objetivo es ayudar al usuario a resolver su problema de software basándote EXCLUSIVAMENTE en el historial de tickets previos que te proporcionaré como contexto.

Instrucciones:
1. Analiza el problema del usuario.
2. Revisa el historial de tickets para encontrar soluciones relevantes en la sección 'Solución/Comentario'.
3. Si encuentras una solución, explícala paso a paso de manera clara y profesional.
4. Si el problema no está documentado en el contexto, pide cordialmente más detalles o indica que no hay registro previo de ese error específico.

Contexto (Historial de tickets):
{context}

Problema del Usuario:
{prompt}
"""
            # 4. Get Streaming Response
            response = model.generate_content(system_prompt, stream=True)
            
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01) # Small delay for smoother streaming feel
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
