from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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

# Normal output function
def normal_output(prompt,user_input):
    # Get response from LLM
    response = llm.invoke(prompt)
    bot_reply = response.content if hasattr(response, 'content') else str(response)

    # Append bot reply to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

    with st.chat_message("bot"):
        st.markdown(bot_reply)



# Streaming function
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


# Streaming function for chain
def streaming_chain(chain,user_input):
    with st.chat_message("bot"):
        placeholder = st.empty()          # ← this will be updated as chunks arrive
        full_text = ""
        
        time.sleep(0.01)

        for chunk in chain.stream({"user_request": user_input}):  
            full_text += chunk.content
            placeholder.markdown(full_text)

        # After streaming, store the full reply in history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "bot", "content": full_text})



#------------------UI-------------------------------------------
st.set_page_config(page_title="The Analysis Bot", page_icon="🔍")
st.title("🔍📈 The Analysis Bot")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

if "filename" not in st.session_state:
    st.session_state.filename = ""

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.4  # Default temperature

# ---- TEMPERATURE SLIDER ----
st.session_state.temperature = st.slider(
    "🌡️ Model Temperature (creativity)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,  # 🔹 Default temperature
    step=0.1,
    help="Lower = more focused/deterministic, Higher = more creative responses."
)


#-----------------------------------modelSelection---------------------------

models = {
    1: "tngtech/deepseek-r1t2-chimera:free",
    2: "z-ai/glm-4.5-air:free",
    3: "google/gemini-2.0-flash-exp:free",
    4: "deepseek/deepseek-r1-0528:free",
    5: "openai/gpt-oss-20b:free"
}

n = st.selectbox("Select a model:", options=list(models.keys()), format_func=lambda x: models[x])

load_dotenv()
api_key = os.getenv(f"OPENAI_API_KEY_{n}")  # Will fetch OPENAI_API_KEY_1, _2, or _3


#-----------------modelInitiliazation
client = httpx.Client(verify=False)

llm = ChatOpenAI(
base_url="https://openrouter.ai/api/v1",
model = models.get(n),
openai_api_key=api_key, 
http_client =  client,
temperature=st.session_state.temperature,
streaming=True,
)



st.markdown("""
    <style>
        /* Global background color */
        body {
            background-color: #ADD8E6; /* Change this color to your desired background color */
            font-family: 'Poppins', sans-serif;
            margin: 0; /* Ensures there's no default margin */
            height: 100vh; /* Ensures full height is covered */
        }
    
            
        /* Upload box */
        .stFileUploader {
            border: 2px dashed #6c63ff !important;
            border-radius: 15px !important;
            padding: 1rem;
            background-color: #f8f9ff;
        }

        /* Text input */
        .stTextInput>div>div>input {
            border: 2px solid #6c63ff !important;
            border-radius: 10px;
            padding: 10px;
            font-size: 1rem;
        }

        /* Button style */
        div.stButton > button:first-child {
            background-color: #6c63ff;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            height: 50px;
            width: 200px;
            border: none;
            transition: 0.3s;
            cursor: pointer;
        }

        div.stButton > button:first-child:hover {
            background-color: #574bff;
            transform: translateY(-3px);
        }
            
    </style>
""", unsafe_allow_html=True)


#------------------chatHistoryInitiliazation
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#--------------------------------------------input------------------------------------------

# File input------------------
if not st.session_state.file_uploaded:
    with st.spinner("Processing document..."):

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
        
            # prompt = f"""
            # You are a helpful analysis chatbot. Review the uploaded file content below and provide a thoughtful summary. 
            # Use your judgment to highlight important insights and offer meaningful interpretations based on the data.

            # Uploaded file content:
            # {st.session_state.file_text}

            # Response:
            # """

            user = """
                Please summarize the uploaded file based strictly on its content.
            """

            system_prompt = f"""
            You are a precise and factual analysis chatbot.

            Your task:
            - Review the uploaded file content provided below.
            - Summarize it accurately.
            - Highlight key insights, patterns, and important points.
            - Do **not** include any information that is not present in the document.
            - If you are uncertain about something, respond with: "The information is not available in the uploaded file."

            Uploaded file content:
            {st.session_state.file_text}
            """

            # Create structured prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{user_request}")
                ]
            )
            chain = prompt | llm

            # bot = chain.invoke({"user_request": user}).content
            # st.markdown(bot)
            streaming_chain(chain, user)

            # Upload button
            st.button('ASK ME')

# After upload: hide uploader, display content
else:
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg", width=80)
        st.success(f"File '{st.session_state.filename}' uploaded and added to context.")
        st.markdown("**Content of uploaded file:**")
        st.text_area("File content", st.session_state.file_text, height=200)



    #text input------------------------
    user_input = st.chat_input("Ask something...")



    #-----------------promtingLogic------------------------------------
    with st.spinner("Thinking..."):
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
        
            # Build prompt with previous chat
            previous_chat = ""
            for msg in st.session_state.chat_history[:-1]:  # exclude current user input
                role = msg["role"]
                content = msg["content"]
                previous_chat += f"{role}: {content}\n"
            
            # Construct the prompt
            prompt = f"""
            You are a helpful and context-aware chatbot. 
            Use the previous conversation to understand the user's intent and respond appropriately.

            Uploaded file content:
            {st.session_state.file_text}

            Previous conversation:
            {previous_chat}
        
            Current user input:
            {user_input}
        
            Respond as the chatbot:
            
            
            note-> user will ask question firstly uploaded document that you have sumraised and then based on that you will answer his question.
            if user ask question related to uploaded document then answer based on that document only otherwise if user ask something else then answer based on your knowledge."""

            # user = """
            # {user_input}
            # """

            # promt = f"""
            # You are a helpful and context-aware chatbot. 
            # Use the previous conversation to understand the user's intent and respond appropriately.

            # Uploaded file content:
            # {st.session_state.file_text}

            # Previous conversation:
            # {previous_chat}

            # Current user input:
            # {user_input}

            # Respond as the chatbot:

            # Note -> The user will ask questions about the uploaded document that you have summarized, and you should answer based on that document.
            # If the user asks a question related to the uploaded document, answer using that document only.
            # Otherwise, if the user asks something else, answer based on your general knowledge.
            # """

            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         ("system", promt),
            #         ("human", user)
            #     ]
            # )

            st.session_state.new_file_uploaded = False

        #--------------------------NormalOutput-----------------------------------------------------------    
            
            #normal_output(prompt,user_input)

        #-------------------------------STREAMING_output----------------------------------------------------------
            streaming(prompt,user_input)
        #-----------------------------------------END-------------------------------------------------------------


        
    

        
#debuging
#print(st.session_state.chat_history)
print("\n 1",st.session_state.temperature)
#print(st.session_state.file_text)