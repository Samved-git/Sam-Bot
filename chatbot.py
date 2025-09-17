import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os

# Set your GOOGLE_API_KEY environment variable (typically set outside script or via Streamlit secrets)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Initialize Google Generative AI model with LangChain
llm = ChatGoogleGenerativeAI(model="gemini-pro")  # or use "gemini-2.5-flash" etc.

# Create conversation chain with memory and LLM
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

st.title("🗣️ Conversational Chatbot with Google Generative AI")
st.subheader("Simple Chat Interface for LLMs by Build Fast with AI")

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
