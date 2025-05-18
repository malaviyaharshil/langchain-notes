from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnableBranch,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
parser = StrOutputParser()

template_report  = PromptTemplate(
    template='give me report on {topic} >500 words',
    input_variables=['report']
)
template_summary = PromptTemplate(
    template='gime me summary of {report}',
    input_variables=['report']
)

chain_report = RunnableSequence(template_report,model,parser)
chain_summary = RunnableSequence(template_summary,model,parser)
chain_mid = RunnableBranch(
    (RunnableLambda(lambda x:len(x.split())>500),chain_summary),
    # RunnableLambda(lambda x: RunnablePassthrough())
    RunnablePassthrough()
)

chain = chain_report | chain_mid
result = chain.invoke({'topic':'ai'})
print(result)
print(len(result.split()))