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

def call_conversation_predict(prompt, retries=3, delay=15):
    for attempt in range(retries):
        try:
            return conversation.predict(input=prompt)
        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower() or "429" in msg:
                if attempt < retries - 1:
                    st.warning(f"Quota exceeded, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
            raise e
    raise RuntimeError("Exceeded maximum retries due to quota limits.")

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            response = call_conversation_predict(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"API call error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, reached API quota limits or an error occurred."})
