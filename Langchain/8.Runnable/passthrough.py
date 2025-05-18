from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model1 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
model2 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
parser = StrOutputParser()
passthrough = RunnablePassthrough()

template_joke = PromptTemplate(
    template='tell me joke in {topic}',
    input_variables=['topic']
)
template_explain = PromptTemplate(
    template='explain the following joke.\n {joke}',
    input_variables=['joke']
)

chain_joke = RunnableSequence(template_joke,model1,parser)
chain_explain = RunnableSequence(template_explain,model2,parser)

chain_mid = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':chain_explain
})

chain = chain_joke | chain_mid
result = chain.invoke({'topic':'ai'})
print(result)