from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=1)
parser = StrOutputParser()

template = PromptTemplate(
    template='Give me 3 facts about {topic}',
    input_variables=['topic']
)

chain = template | model | parser
result = chain.invoke({'topic':'ai'})
print(result)

chain.get_graph().print_ascii()