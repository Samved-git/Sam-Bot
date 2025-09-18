import time
import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from PIL import Image
import pandas as pd
import io
import os

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# UI Styling - light coding theme
st.markdown(
    """
    <style>
    .appview-container, .main, [data-testid="stAppViewContainer"] {
        background-color: #f9fafb !important;
        color: #333333 !important;
        font-family: Consolas, "Courier New", monospace !important;
        min-height: 100vh;
    }
    .st-chat-message {
        background-color: #e1e4e8 !important;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        color: #222222 !important;
    }
    div.st-chat-message[data-builtin-role="user"] {
        background-color: #d1e7ff !important;
        color: #003366 !important;
    }
    div.st-chat-message[data-builtin-role="assistant"] {
        background-color: #a8c0ff !important;
        color: #002244 !important;
    }
    div[data-baseweb="input"] > input {
        background-color: #ffffff !important;
        color: #333333 !important;
        border-radius: 6px !important;
        border: 1px solid #cccccc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
MAX_INPUT_LENGTH = 200

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=2, return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "attached_content" not in st.session_state:
    st.session_state.attached_content = ""
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "uploaded_file_type" not in st.session_state:
    st.session_state.uploaded_file_type = None

st.title("🗣️ Conversational Chatbot sam-bot")
st.subheader("AI Chatbot")

col1, col2 = st.columns([9, 1], gap="small")
with col1:
    user_prompt = st.chat_input("Your question")
with col2:
    open_upload = st.button("📎", help="Attach file", use_container_width=True)

if open_upload:
    uploaded_file = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg", "pdf", "csv", "txt"],
        label_visibility="hidden",
        key="file_upload_modal"
    )
    if uploaded_file is not None:
        st.session_state.uploaded_file_bytes = uploaded_file.read()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_type = uploaded_file.type
        # Rewind for repeated reads
        uploaded_file = io.BytesIO(st.session_state.uploaded_file_bytes)
        file_type = st.session_state.uploaded_file_type
        if file_type.startswith("image/"):
            st.session_state.attached_content = "Image file attached: " + st.session_state.uploaded_file_name
        elif file_type == "application/pdf" and PyPDF2 is not None:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            st.session_state.attached_content = text.strip()[:2000]
        elif file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.session_state.attached_content = df.to_string(index=False)[:2000]
        elif file_type == "text/plain":
            st.session_state.attached_content = uploaded_file.read().decode("utf-8")[:2000]
        else:
            st.session_state.attached_content = "Unsupported file type."

# Display uploaded file persistently
if st.session_state.uploaded_file_bytes is not None:
    uploaded_file = io.BytesIO(st.session_state.uploaded_file_bytes)
    st.info(f"Attached file: {st.session_state.uploaded_file_name}")
    if st.session_state.uploaded_file_type.startswith("image/"):
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Preview: {st.session_state.uploaded_file_name}", use_column_width=True)
    elif st.session_state.uploaded_file_type == "application/pdf":
        st.write("PDF content included in chat context.")
    elif st.session_state.uploaded_file_type == "text/csv":
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        st.write(df)
    elif st.session_state.uploaded_file_type == "text/plain":
        # text already stored in attached_content
        st.write(st.session_state.attached_content)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", location="global")
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

def safe_predict(prompt, max_attempts=1, delay=15):
    context = st.session_state.attached_content
    if context:
        full_prompt = f"The following is the content of the attached file:\n{context}\n\nUser question: {prompt}"
    else:
        full_prompt = prompt
    for attempt in range(max_attempts):
        try:
            return conversation.predict(input=full_prompt)
        except Exception as e:
            error_msg = str(e).lower()
            if ("quota" in error_msg or "429" in error_msg) and attempt < max_attempts - 1:
                st.warning(f"Quota reached, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return None
    return None

if user_prompt:
    user_prompt = user_prompt.strip()
    if len(user_prompt) > MAX_INPUT_LENGTH:
        user_prompt = user_prompt[:MAX_INPUT_LENGTH] + "..."
    if user_prompt in st.session_state.cache:
        cached_response = st.session_state.cache[user_prompt]
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": cached_response})
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.messages[-1]["role"] != "assistant" and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        response = safe_predict(prompt)
        if response:
            st.write(response)
            st.session_state.cache[prompt] = response
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            error_msg = "API quota exceeded or error. Please try again later."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
