import tiktoken
import os
import pinecone
import constants
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.vectorstores import Pinecone


os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY

def load_pinecone(texts,metadata):
    pinecone.init(
    api_key=constants.PINECONE_API_KEY,
    environment=constants.PINECONE_ENV
    )

    if constants.PINECONE_INDEX_NAME not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=constants.PINECONE_INDEX_NAME,
            dimension=1536 
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")    

    docs = []
    for i in range(len(texts)):
        doc = Document(
        page_content=texts[i],
        metadata=metadata[i]
        )
        docs.append(doc)

    return Pinecone.from_documents(docs, embeddings, index_name=constants.PINECONE_INDEX_NAME)    
