import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # Corregido
import re
import base64

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Ing. Custodio - UCE",
    page_icon="🦅",
    layout="wide"
)

if not api_key:
    st.error("❌ ERROR: No encontré la API Key. Revisa tu archivo .env")
    st.stop()

genai.configure(api_key=api_key)

PDF_FOLDER = 'archivos_pdf'
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

LOGO_URL = "UCELOGO.png"
AVATAR_URL = "Custodio.png"

# --- 2. FUNCIONES DE LÓGICA ---

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return ""

def conseguir_modelo_disponible():
    try:
        modelos = list(genai.list_models())
        modelos_chat = [m for m in modelos if 'generateContent' in m.supported_generation_methods]
        if not modelos_chat: return None, "Sin modelos compatibles."
        nombres = [m.name for m in modelos_chat]
        preferidos = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro']
        for pref in preferidos:
            if pref in nombres: return pref, pref
        return nombres[0], nombres[0]
    except Exception as e:
        return None, str(e)

def guardar_archivo(uploaded_file):
    ruta = os.path.join(PDF_FOLDER, uploaded_file.name)
    with open(ruta, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

def eliminar_archivo(nombre_archivo):
    ruta = os.path.join(PDF_FOLDER, nombre_archivo)
    if os.path.exists(ruta): os.remove(ruta)

@st.cache_resource
def leer_pdfs_locales():
    textos, fuentes = [], []
    if not os.path.exists(PDF_FOLDER): return [], []
    archivos = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    for archivo in archivos:
        try:
            ruta_completa = os.path.join(PDF_FOLDER, archivo)
            reader = PyPDF2.PdfReader(ruta_completa)
            for i, page in enumerate(reader.pages):
                texto = page.extract_text()
                if texto:
                    texto_limpio = re.sub(r'\s+', ' ', texto).strip()
                    chunks = [texto_limpio[i:i+1000] for i in range(0, len(texto_limpio), 800)]
                    for chunk in chunks:
                        textos.append(chunk)
                        fuentes.append(f"{archivo} (Pág {i+1})")
        except: pass
    return textos, fuentes

def buscar_informacion(pregunta, textos, fuentes):
    if not textos: return ""
    try:
        vectorizer = TfidfVectorizer().fit_transform(textos + [pregunta])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1]).flatten()
        indices = cosine_sim.argsort()[:-5:-1]
        contexto = ""
        hay_relevancia = False
        for i in indices:
            if cosine_sim[i] > 0.15:
                hay_relevancia = True
                contexto += f"\n- {textos[i]} [Fuente: {fuentes[i]}]\n"
        return contexto if hay_relevancia else ""
    except: return ""

# --- 3. DISEÑO VISUAL ---

def estilos_globales():
    estilos = """
    <style>
        .block-container { padding-top: 4rem !important; padding-bottom: 0rem !important; }
        .footer-credits {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: #ffffff; text-align: center;
            font-size: 11px; padding: 5px; border-top: 2px solid #C59200; z-index: 99999;
        }
    </style>
    <div class="footer-credits">
        <div style="font-weight: bold; color: #002F6C;">Hecho por: Andrango Bryan, Calero Adrián, Flores Ney, Mancero Juan.</div>
        <div style="font-size: 9px; color: #666;">Proyecto Académico | Powered by Google Gemini API</div>
    </div>
    """
    st.markdown(estilos, unsafe_allow_html=True)

# --- 4. INTERFACES ---

def sidebar_uce():
    with st.sidebar:
        st.markdown("### UCE - FICA")
        st.divider()
        st.title("Navegación")
        opcion = st.radio("Ir a:", ["💬 Chat con Ing. Custodio", "📂 Gestión de Bibliografía"])
        return opcion

def interfaz_gestor_archivos():
    estilos_globales()
    st.header("📂 Gestión de Bibliografía")
    uploaded_files = st.file_uploader("Cargar documentos PDF", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Procesar Documentos", type="primary"):
            for file in uploaded_files: guardar_archivo(file)
            leer_pdfs_locales.clear()
            st.success("✅ Documentos aprendidos.")
            st.rerun()
    
    archivos = os.listdir(PDF_FOLDER)
    if archivos:
        for f in archivos:
            c1, c2 = st.columns([4, 1])
            c1.text(f"📄 {f}")
            if c2.button("🗑️", key=f):
                eliminar_archivo(f)
                leer_pdfs_locales.clear()
                st.rerun()

def interfaz_chat():
    estilos_globales()
    col_izquierda, col_derecha = st.columns([1.5, 3])
    
    with col_izquierda:
        if os.path.exists(AVATAR_URL):
            img_b64 = get_img_as_base64(AVATAR_URL)
            st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 65vh;">
                    <img src="data:image/png;base64,{img_b64}" style="width: 100%; max-width: 450px; border-radius: 20px;">
                </div>
            """, unsafe_allow_html=True)

    with col_derecha:
        col_hl, col_ht = st.columns([1.2, 4]) 
        with col_hl:
            if os.path.exists(LOGO_URL):
                st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
                st.image(LOGO_URL, width=150) 
                st.markdown('</div>', unsafe_allow_html=True)
        with col_ht:
            st.markdown("<div style='padding-top: 35px;'><h2 style='margin-bottom: 0px; color: #002F6C;'>💬 Asistente Virtual</h2><p style='margin-top: 0px; color: gray;'>Ing. Custodio - Tu Tutor Virtual de la FICA</p></div>", unsafe_allow_html=True)
        
        st.info("🦅 **¡Hola compañero! Soy el Ing. Custodio.** Escribe tu duda abajo.")
        
        contenedor_chat = st.container(height=300, border=True)

        # Lógica de Chat indentada correctamente dentro de col_derecha
        modelo, status = conseguir_modelo_disponible()
        if not modelo:
            st.error(f"Error: {status}")
            return

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with contenedor_chat:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar=AVATAR_URL if message["role"]=="assistant" else "👤"):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Escribe tu consulta aquí..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            prompt_actual = st.session_state.messages[-1]["content"]
            with contenedor_chat:
                with st.chat_message("assistant", avatar=AVATAR_URL):
                    placeholder = st.empty()
                    try:
                        textos, fuentes = leer_pdfs_locales()
                        contexto = buscar_informacion(prompt_actual, textos, fuentes)
                        model = genai.GenerativeModel(modelo)
                        res = model.generate_content(f"Eres el Ing. Custodio. Contexto: {contexto}. Pregunta: {prompt_actual}")
                        placeholder.markdown(res.text)
                        st.session_state.messages.append({"role": "assistant", "content": res.text})
                    except Exception as e:
                        st.error(f"Error: {e}")

def main():
    opcion = sidebar_uce()
    if opcion == "📂 Gestión de Bibliografía":
        interfaz_gestor_archivos()
    else:
        interfaz_chat()

if __name__ == "__main__":
    main()
