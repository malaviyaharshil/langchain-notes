from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b')

chat_template = ChatPromptTemplate([
    # # this will not work in ChatPromptTemplate
    # SystemMessage(content='You are expert in {domain}'),
    # HumanMessage(content='Explain in brief what is {topic}')
    
   ('system','You are expert in {domain}'),
   ('human','Explain in brief what is {topic}')
])

prompt=chat_template.invoke({'domain':'ai','topic':'nlp'})
print(prompt)


