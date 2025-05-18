from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-sonnet-20240229')

result=model.invoke('hii')

print(result)
