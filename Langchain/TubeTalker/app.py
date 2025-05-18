from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
os.environ["PYTHONPATH"] = ""
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

load_dotenv()

st.set_page_config(page_title="Tube Talker")
st.title('🎬 Tube Talker - Chat with YouTube Videos')

video_id = st.text_input('Enter YouTube Video ID (e.g., dQw4w9WgXcQ):', max_chars=20)

query = st.text_input('Ask a question about the video:', max_chars=100)

if video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcripts = ' '.join(chunk['text'] for chunk in transcript_list)

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.create_documents([transcripts])

        embedding = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        vector_store = FAISS.from_documents(chunks, embedding)

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

        if  st.button("➡️ Submit") and query:
            ret_docs = retriever.invoke(query)
            context = "\n".join(doc.page_content for doc in ret_docs)

            template = PromptTemplate(
                template="""
                    You are an assistant answering user queries based on the context below.
                    If the query is unrelated to the context or the context is insufficient, respond with "I don't know."
                    Context:
                    {context}
                    Query:
                    {query}
                """,
                input_variables=['context', 'query']
            )

            model = GoogleGenerativeAI(model='gemini-1.5-flash-8b', temperature=0.5)
            parser = StrOutputParser()
            chain = template | model | parser
            result = chain.invoke({'context': context, 'query': query})
            st.text_area('Answer:', value=result, height=200)

    except TranscriptsDisabled:
        st.error('Transcripts are disabled or unavailable for this video.')
    except Exception as e:
        st.error(f'An error occurred: {e}')
