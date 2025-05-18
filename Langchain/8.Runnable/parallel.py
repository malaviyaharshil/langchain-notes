from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
model2 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
parser = StrOutputParser()

template_x = PromptTemplate(
    template='write tweet about {topic}',
    input_variables=['topic']
)
template_linkedin = PromptTemplate(
    template='write linked in post  about {topic}',
    input_variables=['topic']
)

chain = RunnableParallel({
    'x':RunnableSequence(template_x,model1,parser),
    'linkedin':RunnableSequence(template_linkedin,model2,parser)
})
result = chain.invoke({'topic':'ai'})
print(result)