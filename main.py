from langchain_openai import ChatOpenAI
import os
import httpx
from dotenv import load_dotenv

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



previous_chat=""
while 1:

    user_input= input("Ask (q to quit):")

    if user_input.lower() == 'q':
        break


    prompt = f"""
        You are a helpful and context-aware chatbot. 
        Use the previous conversation to understand the user's intent and respond appropriately.
    
        Previous conversation:
        {previous_chat}
    
        Current user input:
        {user_input}
    
        Respond as the chatbot:"""
    
    response = llm.invoke(prompt)
    previous_chat+=f""" user : {user_input}
                        chatbot : {response.content}"""

    print(" ")
    print(response.content)



