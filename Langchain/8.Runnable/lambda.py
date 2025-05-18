from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model1 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
model2 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
parser = StrOutputParser()

def word_count(text) :
    return len(text.split())

template_joke =PromptTemplate(
    template='tell me joke in {topic}',
    input_variables=['topic']
)

chain_joke = RunnableSequence(template_joke,model1,parser)
chain_mid = RunnableParallel({
    'joke':RunnablePassthrough(),
    'count':RunnableLambda(word_count),
    # 'count':RunnableLambda(lambda x:len(x.split())),
})
chain = chain_joke|chain_mid
result = chain.invoke({'topic':'ai'})
print(result)