import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

# =========================================================
# CONFIG
# =========================================================
VOICE_ENABLED = False  # 🔒 Voice disabled due to ElevenLabs free-tier limits

# =========================================================
# ENVIRONMENT SETUP
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Veera Enterprise AI", layout="wide")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment / secrets")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# EMBEDDING + FAISS SETUP
# =========================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()
EMBED_DIM = 384

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(EMBED_DIM)

if "documents" not in st.session_state:
    st.session_state.documents = []

# =========================================================
# MAIN UI
# =========================================================
st.title("🚀 Veera Enterprise AI – 4 Agent Stable Demo")

menu = st.sidebar.selectbox(
    "Select Agent",
    [
        "Chat Agent",
        "Document Agent",
        "Voice Agent",
        "Automation Agent"
    ]
)

# =========================================================
# 1️⃣ CHAT AGENT
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
# 2️⃣ DOCUMENT AGENT (FAISS)
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
            st.success("✅ Document indexed successfully")
        else:
            st.warning("⚠ Could not extract text from PDF")

    query = st.text_input("Ask a question from documents:")

    if st.button("Search") and query:
        if not st.session_state.documents:
            st.warning("No documents indexed yet")
        else:
            q_emb = embed_model.encode([query])
            _, idx = st.session_state.faiss_index.search(
                np.array(q_emb).astype("float32"), k=1
            )
            context = st.session_state.documents[idx[0][0]]
            st.subheader("🔎 Relevant Content")
            st.write(context[:600])

# =========================================================
# 3️⃣ VOICE AGENT (TEMPORARILY DISABLED)
# =========================================================
elif menu == "Voice Agent":
    st.header("🔊 Voice Agent")

    st.warning("Voice Agent is temporarily disabled.")
    st.info(
        "Reason: ElevenLabs Free Tier has blocked voice generation due to usage limits.\n\n"
        "This module will be enabled once a paid plan is added or limits are restored."
    )

    st.text_area(
        "Text-to-Speech Preview",
        value="Voice output is currently unavailable.",
        disabled=True
    )

# =========================================================
# 4️⃣ AUTOMATION AGENT
# =========================================================
elif menu == "Automation Agent":
    st.header("⚙ Automation Agent")

    st.write("Automation task simulation")

    if st.button("Send Email"):
        st.success("📧 Email sent successfully (simulated)")

    if st.button("Generate Report"):
        st.info("📄 Report generated successfully (simulated)")

    if st.button("Trigger Task"):
        st.warning("⚙ Background task triggered (simulated)")
