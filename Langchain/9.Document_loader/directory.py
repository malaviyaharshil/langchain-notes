from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
docs = loader.load()
print(len(docs))

# lazy load -> returls only 1 document obj at a time
data = loader.lazy_load() 
for d in data:
    print(d.metadata)