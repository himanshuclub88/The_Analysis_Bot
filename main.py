from langchain_openai import ChatOpenAI
import os
import httpx
from dotenv import load_dotenv

#-----------------modelInitiliazation---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = httpx.Client(verify=False)

llm = ChatOpenAI(
base_url="https://genailab.tcs.in",
model = "azure_ai/genailab-maas-DeepSeek-V3-0324",
openai_api_key=api_key, 
http_client =  httpx.Client(verify=False),
temperature=0.75
)

#-----------------------------------------------------------------

llm.invoke('hi')


