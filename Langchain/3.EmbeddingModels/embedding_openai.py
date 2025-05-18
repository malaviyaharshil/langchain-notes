from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=30)

# for single sentence or word
result_query = embedding.embed_query('Apple is a fruit')

print(result_query)

# for multiple sentences or words
docs = [
    'Apple is fruit',
    'Banana is fruit',
    'Litchi is fruit'
]

result_docs = embedding.embed_documents(docs)

print((result_docs))