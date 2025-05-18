from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel
load_dotenv()

model1 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=0.5)
model2 = GoogleGenerativeAI(model='gemini-1.5-flash-8b',temperature=1)
parser1 = StrOutputParser()
parser2 = StrOutputParser()
parser3 = StrOutputParser()

template_notes = PromptTemplate(
    template='Give me short and simple notes from given text.\n{text}',
    input_variables=['text']
)
template_quiz = PromptTemplate(
    template='Generate 3 quiz questions from given text.\n{text}',
    input_variables=['text']
)
template_final = PromptTemplate(
    template='Combine both notes and quiz and give me in proper format.\n notes:{notes} , quiz:{quiz}',
    input_variables=['notes','quiz']
)

parallel_chain  = RunnableParallel({
    'notes': template_notes | model1 | parser1,
     'quiz': template_quiz | model2 | parser2
})

merge_chain = template_final | model1 | parser3

chain = parallel_chain | merge_chain

text ="""Artificial intelligence (AI) refers to the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.[1] Such machines may be called AIs.

High-profile applications of AI include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); virtual assistants (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, many AI applications are not perceived as AI: "A lot of cutting edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore."[2][3]

Various subfields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, perception, and support for robotics.[a] General intelligence—the ability to complete any task performed by a human on an at least equal level—is among the field's long-term goals.[4] To reach these goals, AI researchers have adapted and integrated a wide range of techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[5]"""
result=chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()