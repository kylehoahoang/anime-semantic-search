import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(page_title="Anime Recommender (Vector Search)", layout="wide")

st.title("Anime Recommender - Step 1: Load & Prepare Data")

csv_path = st.text_input("Path to CSV", "data/anime.csv")

@st.cache_data # Stores result of load_csv into streamlit's cache
def load_csv(path: str):
    df = pd.read_csv(path)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist (failing early with error if needed)
    required = ["anime_id", "Name", "Genres", "Synopsis"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Fill NaNs so string concatenation doesn't produce 'nan'
    for col in ["Name", "English name", "Other name", "Genres", "Synopsis", "Type", "Aired"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # For code cleanliness: process episodes col separate because it might be int and the others are strings
    if "Episodes" in df.columns:
        df["Episodes"] = df["Episodes"].fillna("").astype(str)

    # Build the text that will be embedded later
    df["doc_text"] = (
        df["Name"].str.strip()
        + ". Also known as: "
        + df.get("English name", "").astype(str).str.strip()
        + "; "
        + df.get("Other name", "").astype(str).str.strip()
        + ". Genres: "
        + df["Genres"].str.strip()
        + ". Type: "
        + df.get("Type", "")
        + ". Episodes: "
        + df.get("Episodes", "")
        + ". Synopsis: "
        + df["Synopsis"].str.strip()
    )

    return df

@st.cache_resource #Caches the embedding model across reruns (the model is heavyweight)
def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)

def build_or_load_collection(df: pd.DataFrame, persist_dir: str, collection_name: str):
    client = get_chroma_client(persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    # If already populated, don't re-add on every rerun
    existing = collection.count()
    if existing > 0:
        return collection

    # Ingestion logic if collection is not populated
    embedder = get_embedder("all-MiniLM-L6-v2")

    # Chroma requires unique string ID per item
    # documents: the text
    # converts the columns to a python list to be added to collection and for embedding (documents)
    ids = df["anime_id"].astype(str).tolist()
    documents = df["doc_text"].tolist()

    # Store minimal metadata you want to display later
    # df.iterrows() Might be a little slow but fine for small datasets
    # For faster builds in the future use vectorized pandas operations
    metadatas = []
    # df.iterrows() returns (index, row)
    # the purpose of for _, row is to acknowledge for readability that we are NOT using index
    # in practicality we could do for index, row in df.iterrows() and it'll be the same functionally
    # In the code we only use row because we don't really need to use the index
    for _, row in df.iterrows():
        metadatas.append(
            {
                "Name": row.get("Name", ""),
                "Genres": row.get("Genres", ""),
                "Type": row.get("Type", ""),
                "Score": str(row.get("Score", "")),
            }
        )

    # Encode in batches to avoid high memory usage
    batch_size = 256
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        # embeddings: the vectors (computed from the documents)
        # THIS IS THE MAGIC!
        embeddings = embedder.encode(batch_docs).tolist()

        # add the results to the collection
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
        )

    return collection

try:
    # Load csv and display raw data
    df = load_csv(csv_path)
    st.write("Raw columns:", list(df.columns))

    # Normalize data for nan's
    df2 = normalize_columns(df)

    # Display top 20 of the normalized dataset for debug
    st.subheader("Dataset preview")
    st.dataframe(df2[["anime_id", "Name", "Genres", "Type", "Episodes"]].head(20))

    # Display example text used for embedding debug
    st.subheader("Example `doc_text` (what will be embedded)")
    st.code(df2["doc_text"].iloc[0])

    st.write("Rows:", len(df2))

    # Front end UI to build chroma vector database using our dataset
    st.subheader("Vector index")
    # Text input to ask where to store chromadb
    persist_dir = st.text_input("Chroma persist directory", "chroma_store")
    # Names the collection/bucket inside Chroma where your vectors live
    collection_name = st.text_input("Collection name", "anime")

    # Creates UI button to build collection (runs our function)
    if st.button("Build / Load Vector DB"):
        collection = build_or_load_collection(df2, persist_dir, collection_name)
        st.success(f"Collection ready. Items in collection: {collection.count()}")

    # Query section used to query the vector database
    st.subheader("Search")
    query = st.text_input("Describe what you want to watch")
    k = st.slider("How many recommendations?", 1, 20, 10)
    if st.button("Recommend"):
        # Makes sure collection is loaded first
        collection = build_or_load_collection(df2, persist_dir, collection_name)
        embedder = get_embedder("all-MiniLM-L6-v2")

        # Encodes the query for the model to ingest similar to how we made ingested the data
        # Why the [query] and [0]?
        # encode() is batch-oriented; it returns a list/array of vectors.
        # You pass a list of 1 string, so it returns a list of 1 vector.
        # [0] extracts that single vector.
        q_embedding = embedder.encode([query]).tolist()[0]

        # collection.query performs a nearest-neighbor search in vector space
        # Inputs:
        # query_embeddings=[q_embedding]: list of query vectors (we have 1)
        # n_results=k: how many neighbors to return
        # include=[...]: what extra fields to return
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results = k,
            include=["metadatas", "documents", "distances"],
        )

        # for the selected number of nearest neighbors (results) to return:
        # Since you queried with one embedding, you access the first query’s results with [0].
        # Then [i] picks the i-th neighbor.
        for i in range(k):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] # distance is how close the result is to the query
            doc = results["documents"][0][i]
            # results["metadatas"][0] is a list of length k
            # results["metadatas"][0][i] is the dict for result i
    
            st.markdown(f"### {meta.get('Name', 'Unknown')}  \n**Distance:** `{dist:.4f}`")
            st.write("Genres:", meta.get("Genres", ""))
            st.write("Type:", meta.get("Type", ""), "Score:", meta.get("Score", ""))
            st.caption(doc[:400] + "...")

except Exception as e:
    st.exception(e)