from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()
print(docs)
print(len(docs)) #same as no. of pages
print(docs[0])
print(docs[1].metadata)
