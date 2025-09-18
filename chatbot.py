import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import time
from PIL import Image

# Custom dark background and styling for coding-friendly UI
st.markdown(
    """
    <style>
    /* Full app dark background */
    .appview-container, .main, [data-testid="stAppViewContainer"] {
        background-color: #1e1e1e !important;
        color: #d4d4d4 !important;
        font-family: Consolas, "Courier New", monospace !important;
        min-height: 100vh;
    }
    /* Sidebar dark background */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #252526 !important;
        color: #d4d4d4 !important;
    }
    /* Chat message bubbles */
    .st-chat-message {
        background-color: #2d2d2d !important;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
    }
    div.st-chat-message[data-builtin-role="user"] {
        background-color: #094771 !important;
        color: white !important;
    }
    div.st-chat-message[data-builtin-role="assistant"] {
        background-color: #007acc !important;
        color: white !important;
    }
    /* Input box styling */
    div[data-baseweb="input"] > input {
        background-color: #252526 !important;
        color: #d4d4d4 !important;
        border-radius: 6px !important;
        border: 1px solid #333333 !important;
    }
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #555555;
        border-radius: 20px;
        border: 2px solid #1e1e1e;
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

st.title("🗣️ Conversational Chatbot sam-bot")
st.subheader("AI Chatbot")

col1, col2 = st.columns([9,1], gap="small")
with col1:
    user_prompt = st.chat_input("Your question")
with col2:
    open_upload = st.button("📎", help="Attach file", use_container_width=True)

uploaded_file = None
if open_upload:
    uploaded_file = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg", "pdf", "csv", "txt"],
        label_visibility="hidden",
        key="file_upload_modal"
    )

if uploaded_file is not None:
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
