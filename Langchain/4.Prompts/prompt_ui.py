from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-8b')

# # static prompt
# st.header('Research tool')
# user_input = st.text_input('Enter your prompt')

# if st.button('Summarize'):
#     result = model.invoke(user_input)
#     st.write(result.content)
    
# dynamic prompt

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template1.json')

# # fill the input variables
# prompt=template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })

# if st.button('Summarize'):
#     print(prompt)
#     result = model.invoke(prompt)
#     st.write(result.content)

# using chain PromptTemplate is useful



if st.button('Summarize'):
    chain = template|model
    result = chain.invoke({  
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(result.content)