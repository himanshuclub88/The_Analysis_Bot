from langchain_openai import ChatOpenAI
import os
import httpx
from dotenv import load_dotenv
import streamlit as st
from io import StringIO
from PyPDF2 import PdfReader
import json
import time

#--------------udf-------------------------------------
# Function to extract text from PDF
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def streaming(prompt,user_input):
    with st.chat_message("bot"):
        placeholder = st.empty()          # ← this will be updated as chunks arrive
        full_text = ""
        
        time.sleep(0.01)

        for chunk in llm.stream(prompt):  
            full_text += chunk.content
            placeholder.markdown(full_text)

        # After streaming, store the full reply in history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "bot", "content": full_text})

#-----------------modelSelection---------------------------

n = 3    # can be 1, 2, 3 or 4

load_dotenv()
api_key = os.getenv(f"OPENAI_API_KEY_{n}")  # Will fetch OPENAI_API_KEY_1, _2, or _3

models = {
    1: "deepseek/deepseek-chat-v3.1:free",
    2: "google/gemini-2.5-flash-image-preview",
    3: "openai/gpt-oss-20b:free",
    4: "deepseek/deepseek-chat-v3.1:free"
}


#-----------------modelInitiliazation---------------------------


client = httpx.Client(verify=False)

llm = ChatOpenAI(
base_url="https://openrouter.ai/api/v1",
model = models.get(n),
openai_api_key=api_key, 
http_client =  client,
temperature=0.90,
streaming=True,
)


#------------------UI-------------------------------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot - TCS GenAI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

if "filename" not in st.session_state:
    st.session_state.filename = ""

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#--------------------------------------------input------------------------------------------

# File input------------------
if not st.session_state.file_uploaded:

#File extensions------------------------------------------
    text_extensions = [
    # Plain text
    "txt", "log", "cfg", "ini", "conf", "bat", "cmd", "env",

    # Programming languages
    "py", "java", "c", "cpp", "h", "hpp", "cs", "php", "rb", "go", "rs", "swift", "kt", "m", "scala", "sh", "pl", "r", "ts", "dart",

    # Web files
    "html", "htm", "css", "js", "json", "xml", "yaml", "yml", "md", "jsp", "asp", "aspx", "tsv",

    # Data files
    "csv", "tsv", "sql", "toml", "ini",

    #additional
    "pdf", "json"
    ]

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=text_extensions)
    
    if uploaded_file:
        # PDF/text/JSON file reading logic, always convert to string
        if uploaded_file.type == "application/pdf":
            st.session_state.file_text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/json":
            st.session_state.file_text = json.dumps(json.load(uploaded_file), indent=2)
        else:
            st.session_state.file_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.file_uploaded = True
        st.session_state.filename = uploaded_file.name
    
        prompt = f"""
        You are a helpful analysis chatbot. Review the uploaded file content below and provide a thoughtful summary. 
        Use your judgment to highlight important insights and offer meaningful interpretations based on the data.

        Uploaded file content:
        {st.session_state.file_text}

        Response:
        """

        streaming(prompt,"")
        


        # Inject custom CSS for button styling
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;  /* Green background */
            color: white;               /* White text */
            font-size: 18px;            /* Bigger font size */
            height: 50px;               /* Taller button */
            width: 200px;               /* Fixed width */
            border-radius: 10px;        /* Rounded corners */
            border: black;
        }

        div.stButton > button:first-child:hover {
            background-color: #45a049;  /* Darker green on hover */
        }
        </style>
        """, unsafe_allow_html=True)

        # Add the styled button
        st.button('ASK ME')

# After upload: hide uploader, display content
else:
    with st.sidebar:
        st.success(f"File '{st.session_state.filename}' uploaded and added to context.")
        st.markdown("**Content of uploaded file:**")
        st.text_area("File content", st.session_state.file_text, height=200)



    #text input------------------------
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
            
            # Check if a new file was uploaded to decide prompt logic
            prompt = f"""
            You are a helpful and context-aware chatbot. 
            Use the previous conversation to understand the user's intent and respond appropriately.
        
            Previous conversation:
            {previous_chat}
        
            Current user input:
            {user_input}
        
            Respond as the chatbot:
            
            
            note-> user will ask question firstly uploaded document that you have sumraised and then based on that you will answer his question."""

            st.session_state.new_file_uploaded = False

    #--------------------------NormalOutput-----------------------------------------------------------    
            # # Get response from LLM
            # response = llm.invoke(prompt)
            # bot_reply = response.content if hasattr(response, 'content') else str(response)
        
            # # Append bot reply to chat history
            # st.session_state.chat_history.append({"role": "user", "content": user_input})
            # st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

            # with st.chat_message("bot"):
            #     st.markdown(bot_reply)
            


    #-------------------------------STREAMING_output----------------------------------------------------------
            streaming(prompt,user_input)
    #-----------------------------------------END-------------------------------------------------------------


        
    

        
#debuging
#print(st.session_state.chat_history)
print("\n 1")
print(st.session_state.file_text)