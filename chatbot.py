import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent
import os
import time
import requests

# Load API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

MAX_INPUT_LENGTH = 200

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=2, return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if "cache" not in st.session_state:
    st.session_state.cache = {}

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", location="global")

def fetch_latest_news(topic="technology"):
    # Example fetch from NewsAPI or any public news source (replace with preferred source)
    api_url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&apiKey={st.secrets.get('NEWS_API_KEY', '')}"
    try:
        resp = requests.get(api_url)
        data = resp.json()
        if "articles" in data:
            articles = data["articles"][:3]
            news_summaries = [f"{a['title']} ({a['description']})" for a in articles]
            return "Latest news:\n" + "\n".join(news_summaries)
        else:
            return "No recent news found."
    except Exception as e:
        return f"Error fetching news: {e}"

# Define a LangChain tool for news retrieval
news_tool = Tool(
    name="LatestNewsTool",
    description="Get the latest trending news for any topic.",
    func=fetch_latest_news
)

# Initialize an agent with the tool
agent = initialize_agent([news_tool], llm, memory=st.session_state.buffer_memory, verbose=False)

st.title("🗣️ Conversational Chatbot sam-bot (Live Data)")
st.subheader("AI Chatbot with Real-Time News")

# Upload section unchanged (see previous code)
uploaded_file = st.file_uploader(
    "Upload an image or document (PNG, JPG, PDF, CSV, TXT)",
    type=["png", "jpg", "jpeg", "pdf", "csv", "txt"]
)

# Display uploaded file preview
from PIL import Image
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
            # Use agent for inference, which employs the tool for latest data
            if "latest" in prompt.lower() or "news" in prompt.lower():
                return agent.run(prompt)
            else:
                return llm.predict(prompt)
        except Exception as e:
            error_msg = str(e).lower()
            if ("quota" in error_msg or "429" in error_msg) and attempt < max_attempts - 1:
                st.warning(f"Quota reached, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return None
    return None

if user_prompt := st.chat_input("Your question"):
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

if st.session_state.messages[-1]["role"] != "assistant":
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
            st.session_state.messages.append({"role": "assistant", "content": error_msg)
