from langchain_openai import ChatOpenAI
import os
import httpx
from dotenv import load_dotenv
import streamlit as st
from io import StringIO
from PyPDF2 import PdfReader
import json
import re

#--------------udf-------------------------------------
# Function to extract text from PDF
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

#-----------------modelInitiliazation---------------------------



#-----------------modelInitiliazation---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY_2")  #replace with _n where n=1,2,3 acoording model no

model1 = "deepseek/deepseek-chat-v3.1:free"
model2 = "google/gemini-2.5-flash-image-preview"
model3 = "openai/gpt-oss-20b:free"


client = httpx.Client(verify=False)

llm = ChatOpenAI(
base_url="https://openrouter.ai/api/v1",
model = model2,
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

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File input
with st.sidebar:
    st.header("📂 Upload File (Optional)")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt", "json"])
    
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            st.session_state.file_text = read_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.session_state.file_text = stringio.read()
        elif uploaded_file.type == "application/json":
            st.session_state.file_text = json.load(uploaded_file)
        else:
            st.error("Unsupported file type")

        st.success("✅ File uploaded and content added to context.")
        st.markdown("**Content of uploaded file:**")
        st.text_area("File content", st.session_state.file_text, height=300)


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
    
        Content from uploaded file (if any):
        {st.session_state.file_text}
    
        Current user input:
        {user_input}
    
        Respond as the chatbot:"""
    
        # Get response from LLM
        response = llm.invoke(prompt)
        bot_reply = response.content if hasattr(response, 'content') else str(response)
    
        # Append bot reply to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "bot", "content": bot_reply})
        
        #using image handler
        with st.chat_message("bot"):
            # Extract potential image URLs from bot_reply using regex
            image_matches = re.findall(r'!\[([^\]]*)\]\((https?://[^\s)]+)\)', bot_reply)

            if image_matches:
                # Split text and images for better rendering
                # Remove image Markdown from text to avoid double-rendering
                clean_text = re.sub(r'!\[([^\]]*)\]\((https?://[^\s)]+)\)', '', bot_reply).strip()

                if clean_text:
                    st.markdown(clean_text, unsafe_allow_html=True)  # Allow HTML/Markdown with images removed

                # Render each extracted image
                for alt_text, image_url in image_matches:
                    try:
                        # Use st.image for better control (resizes, handles errors)
                        st.image(image_url, caption=alt_text or "Generated Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to load image from {image_url}: {e}")
                        # Fallback: Try embedding in Markdown
                        st.markdown(f"![{alt_text}]({image_url})")
            else:
                # No images: Just render full Markdown
                st.markdown(bot_reply, unsafe_allow_html=True)
    

        # with st.chat_message("bot"):
        #     st.markdown(bot_reply)
    

        
        #debuging
        print(st.session_state.chat_history)
        print("\n")