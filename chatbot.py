import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import timeimport streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import time
from PIL import Image
import pandas as pd

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
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# --- UI, theme ---
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
    """, unsafe_allow_html=True
)

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
        st.session_state.uploaded_file = uploaded_file
        # Handle file and extract context for chat
        file_type = uploaded_file.type
        attached_content = ""
        if file_type.startswith("image/"):
            st.session_state.attached_content = "Image file attached: " + uploaded_file.name
        elif file_type == "application/pdf":
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            st.session_state.attached_content = text.strip()[:2000]  # Max length
        elif file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.session_state.attached_content = df.to_string(index=False)[:2000]
        elif file_type == "text/plain":
            st.session_state.attached_content = uploaded_file.read().decode("utf-8")[:2000]
        else:
            st.session_state.attached_content = "Unsupported file type."

# Always show attached file and info
if st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
    st.info(f"Attached file: {uploaded_file.name}")
    if uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
    elif uploaded_file.type == "application/pdf":
        st.write("PDF content used for answering questions.")
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.write(df)
    elif uploaded_file.type == "text/plain":
        st.write(st.session_state.attached_content)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", location="global")
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

def safe_predict(prompt, max_attempts=1, delay=15):
    # Append file context to prompt if available
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

from PIL import Image

# Custom light and clean theme with good contrast for coding
st.markdown(
    """
    <style>
    /* Light background and text colors */
    .appview-container, .main, [data-testid="stAppViewContainer"] {
        background-color: #f9fafb !important; /* very light gray */
        color: #333333 !important;
        font-family: Consolas, "Courier New", monospace !important;
        min-height: 100vh;
    }
    /* Sidebar light background */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    /* Chat message bubbles */
    .st-chat-message {
        background-color: #e1e4e8 !important; /* light blue-gray */
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        color: #222222 !important;
    }
    div.st-chat-message[data-builtin-role="user"] {
        background-color: #d1e7ff !important; /* lighter blue */
        color: #003366 !important;
    }
    div.st-chat-message[data-builtin-role="assistant"] {
        background-color: #a8c0ff !important; /* soft blue */
        color: #002244 !important;
    }
    /* Chat input box styling */
    div[data-baseweb="input"] > input {
        background-color: #ffffff !important;
        color: #333333 !important;
        border-radius: 6px !important;
        border: 1px solid #cccccc !important;
    }
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f9fafb;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #b0b0b0;
        border-radius: 20px;
        border: 2px solid #f9fafb;
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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", location="global")
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

st.title("🗣️ Conversational Chatbot sam-bot")
st.subheader("AI Chatbot")

col1, col2 = st.columns([9,1], gap="small")
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
        st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
    st.info(f"Uploaded file: {uploaded_file.name}")
    if uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
    elif uploaded_file.type == "application/pdf":
        st.write("PDF uploaded.")
    elif uploaded_file.type == "text/csv":
        import pandas as pd
        data = pd.read_csv(uploaded_file)
        st.write(data)
    elif uploaded_file.type == "text/plain":
        st.write(uploaded_file.read().decode("utf-8"))

def safe_predict(prompt, max_attempts=1, delay=15):
    for attempt in range(max_attempts):
        try:
            return conversation.predict(input=prompt)
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
