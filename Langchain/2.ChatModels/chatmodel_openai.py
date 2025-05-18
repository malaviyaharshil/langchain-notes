from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4',temprature=1.2,max_completion_tokens=10)
result = model.invoke('hii')

print(result)