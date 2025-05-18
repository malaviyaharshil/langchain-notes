from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.7)

schema = [
    ResponseSchema(name='fact_1',description='fact 1 of topic'),
    ResponseSchema(name='fact_2',description='fact 2 of topic'),
    ResponseSchema(name='fact_3',description='fact 3 of topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me facts about {topic}.\n{format}',
    input_variables=['topic'],
    partial_variables={'format':parser.get_format_instructions()}
)
prompt = template.invoke({'topic':'ai'})
result = model.invoke(prompt)
print(result)


# using chain

chain = template | model | parser
chain_result = chain.invoke({'topic':'ai'})
print(chain_result)