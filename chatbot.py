import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import time

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    location="global"
)

conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

st.title("🗣️ Conversational Chatbot sam-bot")
st.subheader("Simple Chat Interface")

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def call_conversation_predict(prompt, max_retries=1, delay=15):
    for attempt in range(max_retries):
        try:
            return conversation.predict(input=prompt)
        except Exception as e:
            msg = str(e).lower()
            if ("quota" in msg or "429" in msg) and attempt < max_retries - 1:
                st.warning(f"Quota exceeded, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                # Stop retrying if quota still exceeded or other error
                raise e
    return None

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            response = call_conversation_predict(prompt)
            if response:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                msg = "API quota exceeded, please try again later."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
        except Exception as e:
            st.error(f"API call error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, something went wrong."})
