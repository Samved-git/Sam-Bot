import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import time

# Set API key from secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize session state for memory & messages
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Initialize Google GenAI model with a supported model name
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", location="global")

# Create conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

st.title("🗣️ Conversational Chatbot sam-bot")
st.subheader("Optimized for performance and quota handling")

# User input
if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Function to call API with limited retries
def safe_predict(prompt, max_attempts=1, delay=15):
    for attempt in range(max_attempts):
        try:
            return conversation.predict(input=prompt)
        except Exception as e:
            error_msg = str(e).lower()
            if ("quota" in error_msg or "429" in error_msg) and attempt < max_attempts - 1:
                st.warning(f"Quota reached, wait {delay} seconds and retry...")
                time.sleep(delay)
            else:
                return None
    return None

# Generate response only if last message is user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"), st.spinner("Processing..."):
        response = safe_predict(prompt)
        if response:
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            msg = "API quota exceeded or error. Please try again later."
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

