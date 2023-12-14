from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
import key_param

# Prepare the collection
client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

# Load the files
loader = DirectoryLoader( './sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

# Generate the embeddings
embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

# Conduct vector search
vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)
