import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

FOLDER_PATH = "docs"
DB_PATH = "./chroma_db"


def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({
                    "filename": file,
                    "content": text.strip().replace("\n", " ")
                })
    return pd.DataFrame(docs)


def save_metadata(df, filename="metadata.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4)


def generate_embeddings(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df["embeddings"] = df["content"].apply(lambda x: model.encode(x).tolist())
    return df


def store_in_chroma(df, db_path=DB_PATH):
    client = chromadb.Client(Settings(persist_directory=db_path))
    collection = client.get_or_create_collection("documents")

    for _, row in df.iterrows():
        collection.add(
            documents=[row["content"]],
            embeddings=[row["embeddings"]],
            ids=[row["filename"]]
        )

    print(f"Stored {len(df)} documents in ChromaDB.")


def query_db(query_text, db_path=DB_PATH, top_k=2):
    client = chromadb.Client(Settings(persist_directory=db_path))
    collection = client.get_collection("documents")
    results = collection.query(query_texts=[query_text], n_results=top_k)
    return results


if __name__ == "__main__":
    print("Loading documents...")
    df = load_documents(FOLDER_PATH)

    print("Saving metadata...")
    save_metadata(df)

    print("Generating embeddings...")
    df = generate_embeddings(df)

    print("Saving into ChromaDB...")
    store_in_chroma(df)

    print("Running a sample query...")
    query = "machine learning applications"
    results = query_db(query)
    print(f"Results for '{query}':\n", results)
