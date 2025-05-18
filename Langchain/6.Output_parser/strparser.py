from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b')

# prompt -1 
template1 = PromptTemplate(
    template='Write Detailed report of {topic}',
    input_variables=['topic']
)
# prompt 02
template2 = PromptTemplate(
    template='Write 5 line summary of foloowing text./n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'AI'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1})
result2 = model.invoke(prompt2)

print(result2)

# using parser
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result_parser=chain.invoke({'topic':'AI'})

print(result_parser)
