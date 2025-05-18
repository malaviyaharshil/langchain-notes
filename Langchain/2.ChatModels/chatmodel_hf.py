from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

HF_Token=os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"]
llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation'

)
model= ChatHuggingFace(llm=llm)

result=model.invoke('What is capital of india?')
print(result)