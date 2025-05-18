from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b')
chat_template = ChatPromptTemplate([
    ('system','you are customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
# load chat history
with open('chat_history.txt') as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    'chat_history':chat_history,
    'query':'where is my refund?'
})

print(model.invoke(prompt))