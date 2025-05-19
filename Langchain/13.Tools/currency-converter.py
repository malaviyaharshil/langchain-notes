from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import os
import requests
import json

llm = ChatOllama(model='llama3.2:3b')
messages = [HumanMessage('convert 5 usd to inr')]

api_key=os.getenv('EXCHANGERATE_KEY')
@tool
def get_conversion_factor(base_currency:str,target_currency:str) -> float:
    """this function fetches currency conversion factor between base and target currency"""
    url=f'https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

# print(get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'}))

@tool
def convert(base_currency_value:float,conversion_rate:Annotated[float,InjectedToolArg]) -> float: #llm will not set value of convertion_rate
    """calculate target currency value based on currency convertion rate"""
    return base_currency_value*conversion_rate

llm_with_tools=llm.bind_tools([get_conversion_factor,convert])
ai_message=llm_with_tools.invoke(messages)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    if tool_call['name'] =='get_conversion_factor':
        response1 = get_conversion_factor.invoke(tool_call)
        messages.append(response1)
        conversion_rate=json.loads(response1.content)['conversion_rate']
    
    if tool_call['name']=='convert':
        tool_call['args']['conversion_rate']=conversion_rate
        response2 = convert.invoke(tool_call)
        messages.append(response2) 


print(llm_with_tools.invoke(messages).content)