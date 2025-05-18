from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.7)
parser  = JsonOutputParser()

template = PromptTemplate(
    template='give me name, age, city of fiction character.\n{format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt= template.format()

result= model.invoke(prompt)
print(result)

parsed_output = parser.parse(result)
print(parsed_output)

# using chain

chain = template | model | parser
chain_result = chain.invoke({})

print(chain_result)