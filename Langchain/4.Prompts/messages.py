
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b')

messages=[
    SystemMessage(content='You are helpful assistant'),
    HumanMessage('Tell me about ai')
]

messages.append(AIMessage(content=model.invoke(messages).content))

print(messages)
