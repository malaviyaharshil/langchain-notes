from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model1 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
model2 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)

class Sentiment(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description='sentiment of feedbach')
    
parser1 = PydanticOutputParser(pydantic_object=Sentiment)
parser2 = StrOutputParser()

template_sentiment = PromptTemplate(
    template='Classify given feedback into positive or negative sentiment.\n {feedback} \n{format}',
    input_variables=['feedback'],
    partial_variables={'format':parser1.get_format_instructions()}
)
template_positive = PromptTemplate(
    template='give reply based in this positive feedback.\n {feedback} ',
    input_variables=['feedback']
)
template_negative = PromptTemplate(
    template='give reply based in this negative feedback.\n {feedback} ',
    input_variables=['feedback']
)

classifier_chain = template_sentiment | model1 | parser1
chain_pos = template_positive | model1 | parser2
chain_neg = template_negative | model2 | parser2
feedback = 'product was amazing.Worth it'
# feedback = 'Quality was not that much worth of money'

result = classifier_chain.invoke({'feedback':feedback})
print(result.sentiment)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive',chain_pos),
    (lambda x:x.sentiment=='negative',chain_neg),
    RunnableLambda(lambda x:'could not fine sentiment')
)

chain = classifier_chain | branch_chain

reply=chain.invoke({'feedback':feedback})
print(reply)