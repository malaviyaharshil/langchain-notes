from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=1)
parser = StrOutputParser()

template_report = PromptTemplate(
    template='Generate detailed report on {topic}',
    input_variables=['topic']
)
template_summary = PromptTemplate(
    template='Give important points from {report}',
    input_variables=['report']
)

chain = template_report | model | parser | template_summary | model | parser
result = chain.invoke({"topic":'ai'})
print(result)