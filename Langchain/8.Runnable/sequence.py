from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
parser = StrOutputParser()

template_joke = PromptTemplate(
    template='write joke about {topic}',
    input_variables=['topic']
)
template_explain = PromptTemplate(
    template='write joke about {joke}',
    input_variables=['joke']
)

chain = RunnableSequence(template_joke,model,parser,template_explain,model,parser)
result = chain.invoke({'topic':'ai'})
print(result)