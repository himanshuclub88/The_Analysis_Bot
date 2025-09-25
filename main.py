from langchain_openai import ChatOpenAI
import os
import httpx
from dotenv import load_dotenv
import streamlit as st

#-----------------modelInitiliazation---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = httpx.Client(verify=False)

llm = ChatOpenAI(
base_url="https://openrouter.ai/api/v1",
model = "deepseek/deepseek-chat-v3.1:free",
openai_api_key=api_key, 
http_client =  client,
temperature=0.75
)

#-----------------------------------------------------------------

#------------------UI-------------------------------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot - TCS GenAI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#text input
user_input = st.chat_input("Ask something...")

#-----------------promtingLogic------------------------------------
with st.spinner("Processing document..."):
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
    
        # Build prompt with previous chat
        previous_chat = ""
        for msg in st.session_state.chat_history[:-1]:  # exclude current user input
            role = msg["role"]
            content = msg["content"]
            previous_chat += f"{role}: {content}\n"
        
        prompt = f"""
        You are a helpful and context-aware chatbot. 
        Use the previous conversation to understand the user's intent and respond appropriately.
    
        Previous conversation:
        {previous_chat}
    
        Current user input:
        {user_input}
    
        Respond as the chatbot:"""
    
        # Get response from LLM
        response = llm.invoke(prompt)
        bot_reply = response.content if hasattr(response, 'content') else str(response)
    
        # Append bot reply to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "bot", "content": bot_reply})
        
    
    
        with st.chat_message("bot"):
            st.markdown(bot_reply)
    
        #debuging
        print(st.session_state.chat_history)
        print("\n")