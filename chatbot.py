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

st.title("🗣️ Conversational Chatbot sam-bot")
st.subheader("AI Chatbot")

# Initialize session_state items
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=2, return_messages=True)
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "attached_text" not in st.session_state:
    st.session_state.attached_text = ""

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Layout: input and clip upload button side by side
col1, col2 = st.columns([9, 1], gap="small")

with col1:
    user_prompt = st.chat_input("Your question")
with col2:
    upload_clicked = st.button("📎", help="Attach a file")

uploaded_file = None
if upload_clicked:
    uploaded_file = st.file_uploader(
        label="Upload file",
        type=["png", "jpg", "jpeg", "pdf", "csv", "txt"],
        label_visibility="visible",
        key="file_uploader"
    )
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        # Read and parse content to attached_text
        try:
            if uploaded_file.type.startswith("image/"):
                st.session_state.attached_text = f"An image file named {uploaded_file.name} is uploaded."
            elif uploaded_file.type == "application/pdf" and PyPDF2 is not None:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                st.session_state.attached_text = text[:2000]
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.session_state.attached_text = df.to_string(index=False)[:2000]
            elif uploaded_file.type == "text/plain" or uploaded_file.name.endswith(".txt"):
                uploaded_file.seek(0)
                st.session_state.attached_text = uploaded_file.read().decode("utf-8")[:2000]
            else:
                st.session_state.attached_text = "Unsupported file type."
        except Exception as e:
            st.session_state.attached_text = "Error reading file content."

# Show uploaded file info & preview persistently
if st.session_state.uploaded_file is not None:
    f = st.session_state.uploaded_file
    st.info(f"Attached file: {f.name}")
    try:
        if f.type.startswith("image/"):
            image = Image.open(f)
            st.image(image, caption=f.name, use_column_width=True)
        elif f.type == "application/pdf":
            st.write("PDF document loaded for chat context.")
        elif f.type == "text/csv":
            df = pd.read_csv(f)
            st.write(df)
        elif f.type == "text/plain" or f.name.endswith(".txt"):
            st.write(st.session_state.attached_text)
    except Exception as e:
        st.error(f"Error displaying attached file: {str(e)}")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", location="global")
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

def safe_predict(prompt, max_attempts=1, delay=15):
    full_prompt = prompt
    if st.session_state.attached_text:
        full_prompt = f"Context - attached file content:\n{st.session_state.attached_text}\n\nUser question: {prompt}"

    for attempt in range(max_attempts):
        try:
            return conversation.predict(input=full_prompt)
        except Exception as e:
            if ("quota" in str(e).lower() or "429" in str(e)) and attempt < max_attempts - 1:
                st.warning(f"Quota reached, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return None
    return None

if user_prompt:
    user_prompt = user_prompt.strip()
    if len(user_prompt) > 200:
        user_prompt = user_prompt[:200] + "..."
    if user_prompt in st.session_state.cache:
        response = st.session_state.cache[user_prompt]
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
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
