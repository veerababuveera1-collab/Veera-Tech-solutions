import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from elevenlabs.client import ElevenLabs
from pypdf import PdfReader

# =========================================================
# ENVIRONMENT SETUP
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

st.set_page_config(page_title="Veera Enterprise AI", layout="wide")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)

eleven_client = None
if ELEVENLABS_API_KEY:
    eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# =========================================================
# EMBEDDING + FAISS SETUP
# =========================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()
dimension = 384

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)

if "documents" not in st.session_state:
    st.session_state.documents = []

# =========================================================
# MAIN UI
# =========================================================
st.title("🚀 Veera Enterprise AI – 4 Agent Working Demo")

menu = st.sidebar.selectbox(
    "Select Agent",
    ["Chat Agent", "Document Agent", "Voice Agent", "Automation Agent"]
)

# =========================================================
# 1. CHAT AGENT
# =========================================================
if menu == "Chat Agent":
    st.header("💬 Chat Agent")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:")

    if st.button("Send") and user_input:
        st.session_state.chat_history.append(("user", user_input))

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": user_input}]
        )

        reply = response.choices[0].message.content
        st.session_state.chat_history.append(("assistant", reply))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")

# =========================================================
# 2. DOCUMENT AGENT
# =========================================================
elif menu == "Document Agent":
    st.header("📄 Document Agent")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        reader = PdfReader(uploaded)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content

        if text.strip():
            st.session_state.documents.append(text)
            emb = embed_model.encode([text])
            st.session_state.faiss_index.add(
                np.array(emb).astype("float32")
            )
            st.success("✅ Document indexed!")
        else:
            st.warning("⚠ Could not extract text from PDF.")

    query = st.text_input("Ask about document:")

    if st.button("Search") and query:
        if len(st.session_state.documents) == 0:
            st.warning("No documents indexed.")
        else:
            q_emb = embed_model.encode([query])
            D, I = st.session_state.faiss_index.search(
                np.array(q_emb).astype("float32"), k=1
            )

            context = st.session_state.documents[I[0][0]]
            st.subheader("Relevant document snippet:")
            st.write(context[:500])

# =========================================================
# 3. VOICE AGENT (FINAL CLOUD-SAFE VERSION)
# =========================================================
elif menu == "Voice Agent":
    st.header("🔊 Voice Agent")

    if not eleven_client:
        st.warning("Voice agent disabled. Add ELEVENLABS_API_KEY in .env")
    else:
        text = st.text_input("Enter text to speak:")

        if st.button("Speak") and text:
            try:
                voices = eleven_client.voices.get_all()
                voice_id = voices.voices[0].voice_id

                audio_generator = eleven_client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_monolingual_v2"
                )

                # Convert generator to bytes
                audio_bytes = b""
                for chunk in audio_generator:
                    audio_bytes += chunk

                # Play directly in Streamlit
                st.audio(audio_bytes, format="audio/mp3")

                st.success("Audio generated successfully.")

            except Exception as e:
                st.error(f"Voice error: {e}")

# =========================================================
# 4. AUTOMATION AGENT
# =========================================================
elif menu == "Automation Agent":
    st.header("⚙ Automation Agent")

    st.write("Simple automation task simulation.")

    if st.button("Send Email"):
        st.success("📧 Email sent (simulated)")

    if st.button("Generate Report"):
        st.info("📄 Report generated (simulated)")

    if st.button("Trigger Task"):
        st.warning("⚙ Task triggered (simulated)")
