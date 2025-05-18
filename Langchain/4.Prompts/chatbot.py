from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b')

history=[]
while True:
    user_input= input('You : ')
    history.append(user_input)
    if(user_input=='exit'):
        break
    result = model.invoke(history)
    history.append(result)
    print('AI :',result)