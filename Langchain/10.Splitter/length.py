from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
text ="""AI automates repetitive learning and discovery through data. Instead of automating manual tasks, AI performs frequent, high-volume, computerised tasks. And it does so reliably and without fatigue. Of course, humans are still essential to set up the system and ask the right questions.

AI adds intelligence to existing products. Many products you already use will be improved with AI capabilities, much like Siri was added as a feature to a new generation of Apple products. Automation, conversational platforms, bots and smart machines can be combined with large amounts of data to improve many technologies. Upgrades at home and in the workplace, range from security intelligence and smart cams to investment analysis.

AI adapts through progressive learning algorithms to let the data do the programming. AI finds structure and regularities in data so that algorithms can acquire skills. Just as an algorithm can teach itself to play chess, it can teach itself what product to recommend next online. And the models adapt when given new data. 

AI analyses more and deeper data using neural networks that have many hidden layers. Building a fraud detection system with five hidden layers used to be impossible. All that has changed with incredible computer power and big data. You need lots of data to train deep learning models because they learn directly from the data. """

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
result  = splitter.split_text(text)

print(result)

result_docs = splitter.split_documents(docs)
print(result_docs[0].page_content)