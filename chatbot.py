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

st.set_page_config(page_title="Conversational Chatbot Sam-Bot", layout="wide")

# Session state initialization for persistence
if 'uploaded_file_bytes' not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'uploaded_file_type' not in st.session_state:
    st.session_state.uploaded_file_type = None
if 'attached_content' not in st.session_state:
    st.session_state.attached_content = ""
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=2, return_messages=True)
if 'cache' not in st.session_state:
    st.session_state.cache = {}

# Load API key securely
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

MAX_INPUT_LENGTH = 200

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", location="global")
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

st.title("🗣️ Conversational Chatbot Sam-Bot")
st.subheader("AI Chatbot")

# Layout for input and upload side by side
col1, col2 = st.columns([9, 1], gap="small")

with col1:
    user_prompt = st.chat_input("Your question")

with col2:
    file_upload_clicked = st.button("📎", help="Upload a file")

# Show file uploader only after button click
if file_upload_clicked:
    uploaded_file = st.file_uploader(
        label="Upload an image or document",
        type=["png", "jpg", "jpeg", "pdf", "csv", "txt"],
        key="file_uploader",
        label_visibility="visible"
    )
    if uploaded_file is not None:
        # Save uploaded file bytes and metadata in session state
        st.session_state.uploaded_file_bytes = uploaded_file.read()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_type = uploaded_file.type

        # Extract file content as context
        file_bytes_io = io.BytesIO(st.session_state.uploaded_file_bytes)
        file_type = st.session_state.uploaded_file_type
        try:
            if file_type.startswith("image/"):
                # Just set note for image
                st.session_state.attached_content = f"Image file attached: {st.session_state.uploaded_file_name}"
            elif file_type == "application/pdf" and PyPDF2 is not None:
                pdf_reader = PyPDF2.PdfReader(file_bytes_io)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                st.session_state.attached_content = text.strip()[:2000]
            elif file_type == "text/csv":
                df = pd.read_csv(file_bytes_io)
                st.session_state.attached_content = df.to_string(index=False)[:2000]
            elif file_type == "text/plain" or st.session_state.uploaded_file_name.endswith(".txt"):
                file_bytes_io.seek(0)
                st.session_state.attached_content = file_bytes_io.read().decode("utf-8")[:2000]
            else:
                st.session_state.attached_content = f"Unsupported file type: {file_type}"
        except Exception as e:
            st.error(f"Failed to read file content: {str(e)}")
            st.session_state.attached_content = ""

# Display the uploaded file info and preview persistently
if st.session_state.uploaded_file_bytes is not None:
    file_bytes_io = io.BytesIO(st.session_state.uploaded_file_bytes)
    st.info(f"Attached file: {st.session_state.uploaded_file_name}")

    # Display preview/content depending on type
    if st.session_state.uploaded_file_type.startswith("image/"):
        try:
            image = Image.open(file_bytes_io)
            st.image(image, caption=st.session_state.uploaded_file_name, use_column_width=True)
        except Exception as e_img:
            st.error(f"Error displaying image: {e_img}")
    elif st.session_state.uploaded_file_type == "application/pdf":
        st.write("PDF content loaded and used for answers.")
    elif st.session_state.uploaded_file_type == "text/csv":
        try:
            file_bytes_io.seek(0)
            df = pd.read_csv(file_bytes_io)
            st.write(df)
        except Exception as e_csv:
            st.error(f"Error reading CSV: {e_csv}")
    elif st.session_state.uploaded_file_type == "text/plain" or st.session_state.uploaded_file_name.endswith(".txt"):
        st.write(st.session_state.attached_content)

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
