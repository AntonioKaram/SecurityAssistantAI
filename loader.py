from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

import os
import logging
from pathlib import Path


DATA_DIR = "./data"
EMBED_DELAY = 0.02
BATCH_SIZE = 96
DEFAULT_COLLECTION = "chroma"

def get_txt_files(data_dir=DATA_DIR):
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        yield str(path)

def get_csv_files(data_dir=DATA_DIR):
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        yield str(path)

def load_data(dir=DATA_DIR):
    pdf_loader = PyPDFDirectoryLoader("./data")
    text_loader = TextLoader("./data")
    csv_loader = CSVLoader("./data")

    loaders = [pdf_loader]

    paths = get_txt_files(dir)
    for path in paths:
        loaders.append(TextLoader(path))

    paths = get_csv_files(dir)
    for path in paths:
        loaders.append(CSVLoader(path))


    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=0, 
                        length_function=len,
                        is_separator_regex=False
    )
    
    docs = text_splitter.split_documents(documents)
    print(f"Total Number of Documents: {len(docs)}")

    return docs


def create_vector_store(docs, embeddings=None, collection_name=DEFAULT_COLLECTION):
    if not docs:
        logging.warning("Empty documents provided to vector store")

    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings, 
        persist_directory=os.path.join("./store", collection_name))
    
    db.add_documents(docs)

    return db