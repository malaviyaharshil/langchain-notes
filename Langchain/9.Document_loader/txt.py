from langchain_community.document_loaders import TextLoader 

loader  = TextLoader('a.txt')
docs = loader.load()
print(docs)
print(type(docs))
print(type(docs[0]))
print(docs[0].metadata)

