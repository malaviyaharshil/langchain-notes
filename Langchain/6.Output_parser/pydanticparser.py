from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=1)

class Person(BaseModel):
    name:str = Field(description='name of the person'),
    age : int = Field(gt=18,description='age of the person'),
    city:str = Field(description='city of the person')
    
parser = PydanticOutputParser(pydantic_object=Person)

template  = PromptTemplate(
    template='Give me random name,age and city of person.\n{format}',
    input_variables=[],
    partial_variables={'format':parser.get_format_instructions()}
)
prompt = template.invoke({})
result = model.invoke(prompt)
parsed_result = parser.parse(result)
print(parsed_result)

# using chains

chain = template | model | parser
chain_result=chain.invoke({})

print(chain_result)