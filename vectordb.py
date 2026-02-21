
import os
import kagglehub
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import pandas as pd

LIMIT = 2000  # Limit the number of documents for testing

MODEL = 'qwen3-embedding:0.6b'


def load_dataset():

    path = kagglehub.dataset_download("huhuyngun/vietnamese-chatbot-ver2")

    csv_file = None

    for file in os.listdir(path):
        if file.lower().endswith(".csv"):
            csv_file = os.path.join(path, file)
            break
        else:
            raise FileNotFoundError("No CSV file found in the dataset.")

    df = pd.read_csv(csv_file)

    missing = df.isnull().sum()
    print(f"Missing values in each column:\n{missing}")

    df = df.dropna(subset=["question", "answers"])

    missing = df.isnull().sum()

    print(f'\nChecking data again : \n{missing}')
    return csv_file, df


def process_data(df, limit=LIMIT):

    if limit is not None:
        df = df.head(limit)

    texts = ("Question: " + df["question"].astype(str) +
             " Answer: " + df["answers"].astype(str)).tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)

    documents = splitter.create_documents(
        texts)

    return documents, [doc.page_content for doc in documents]


def create_or_load_collection():

    embed_model = OllamaEmbeddings(model=MODEL)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = chroma_client.get_or_create_collection(
            name="collections", metadata={"hnsw:space": "cosine"})
    except Exception as e:
        print(f"Error accessing collection: {e}")

    return collection, embed_model


def index_data():
    collection, embed_model = create_or_load_collection()

    if collection.count() == 0:
        print("\n\nIndexing data for the first time !\n\n")
        csv_file, df = load_dataset()
        documents, chunks = process_data(df)

        embeddings = embed_model.embed_documents(chunks)

        ids = [f"id{i}" for i in range(len(documents))]
        metadatas = [{"source": csv_file, "chunk_index": i}
                     for i in range(len(documents))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=chunks,
        )
        print("\n\nIndex data completed :D :D\n\n")
    else:
        print("\n\nCollection already indexed. Skipping indexing step.\n\n")
    return collection, embed_model


collection, embed_model = index_data()
