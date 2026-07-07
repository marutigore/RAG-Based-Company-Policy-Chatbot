"""
Retriever Module.
Manages vector storage, indexing, and retrieval using ChromaDB.
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import config
from utils.embedder import get_embedding, get_embeddings_batch

# Initialize module logger
logger = logging.getLogger("utils.retriever")

# Persistent Chroma client reference
_chroma_client: Optional[chromadb.PersistentClient] = None
# Target collection reference
_collection: Optional[chromadb.Collection] = None
COLLECTION_NAME = "company_policies"


def get_db_client() -> chromadb.PersistentClient:
    """
    Initializes and returns the persistent ChromaDB client instance.

    Returns:
        chromadb.PersistentClient: Persistent database client.
    """
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing persistent ChromaDB client at: {config.VECTOR_DB_DIR}")
        
        # Ensure target storage directory exists
        try:
            os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
            # Create persistent client
            _chroma_client = chromadb.PersistentClient(
                path=config.VECTOR_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            logger.error(f"Error creating ChromaDB storage directory: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
            
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """
    Retrieves or creates the database collection.

    Returns:
        chromadb.Collection: Vector DB collection object.
    """
    global _collection
    if _collection is None:
        client = get_db_client()
        try:
            # Get or create collection. We pass None for embedding_function because we supply pre-computed embeddings.
            _collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' successfully resolved.")
        except Exception as e:
            logger.error(f"Error fetching ChromaDB collection: {e}")
            raise RuntimeError(f"Failed to fetch collection: {e}")
            
    return _collection


def add_documents_to_db(chunks: List[Dict[str, Any]]) -> None:
    """
    Embeds a list of document chunks and writes them to the ChromaDB vector index.

    Args:
        chunks (List[Dict[str, Any]]): List of chunked page segments containing 'text' and 'metadata'.

    Raises:
        ValueError: If input is empty or invalid.
        RuntimeError: If database write fails.

    Example:
        >>> add_documents_to_db([{'text': 'Policy detail...', 'metadata': {'source': 'file.pdf'}}])
    """
    if not chunks:
        logger.warning("No chunks provided to insert into vector store.")
        return

    collection = get_collection()
    texts: List[str] = [chunk["text"] for chunk in chunks]
    
    try:
        logger.info(f"Embedding {len(chunks)} chunks for database insertion...")
        # Compute embeddings for all chunks in batches
        embeddings = get_embeddings_batch(texts)
        
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        # Build payload arrays for ChromaDB
        for idx, chunk in enumerate(chunks):
            # Formulate a unique ID based on filename, page, and chunk index
            source = chunk["metadata"].get("source", "unknown_source")
            page = chunk["metadata"].get("page", 0)
            chunk_idx = chunk["metadata"].get("chunk_idx", 0)
            
            unique_id = f"{source}_p{page}_c{chunk_idx}_{idx}"
            ids.append(unique_id)
            
            # Format metadata: ChromaDB requires primitive types (str, int, float, bool)
            meta = {
                "source": str(source),
                "page": int(page),
                "token_count": int(chunk["metadata"].get("token_count", 0))
            }
            metadatas.append(meta)
            documents.append(chunk["text"])

        logger.info(f"Writing {len(ids)} embedded documents to ChromaDB collection...")
        # Insert elements into the collection
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("Database write completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during document addition to database: {e}")
        raise RuntimeError(f"Database insertion failed: {e}")


def query_db(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Queries the vector database for the top-k most similar chunks.

    Args:
        query_text (str): Natural language user question.
        k (int): Number of top results to retrieve.

    Returns:
        List[Dict[str, Any]]: List of matching results, each with 'text' and 'metadata'.

    Example:
        >>> matches = query_db("What is the sick leave policy?", k=3)
        >>> print(matches[0]['metadata']['source'])
        "hr_policy.pdf"
    """
    if not query_text or not query_text.strip():
        logger.warning("Empty query submitted to retriever.")
        return []

    collection = get_collection()

    try:
        logger.info(f"Retrieving top {k} contexts for query: '{query_text}'")
        # Generate embedding for search query
        query_vector = get_embedding(query_text)
        
        # Query collection using pre-computed vector
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        retrieved_items: List[Dict[str, Any]] = []
        
        # Parse output from ChromaDB structure
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
            
            for idx in range(len(docs)):
                # Convert distance to a similarity score (cosine distance is 0 to 2, similarity = 1 - distance)
                # Note: exact interpretation depends on HNSW config
                distance = distances[idx]
                similarity = 1.0 - (distance / 2.0)
                
                retrieved_items.append({
                    "text": docs[idx],
                    "metadata": metas[idx],
                    "similarity": round(similarity, 4)
                })
                
        logger.info(f"Retrieved {len(retrieved_items)} results from database.")
        return retrieved_items

    except Exception as e:
        logger.error(f"Error querying vector database: {e}")
        raise RuntimeError(f"Database retrieval failed: {e}")


def reset_db() -> None:
    """
    Resets the persistent vector database by dropping the collection
    and clearing the physical directory contents.
    """
    global _chroma_client, _collection
    logger.warning("Resetting the vector database...")
    
    try:
        # Close handles and delete DB directory
        _collection = None
        _chroma_client = None
        
        if os.path.exists(config.VECTOR_DB_DIR):
            shutil.rmtree(config.VECTOR_DB_DIR)
            logger.info(f"Deleted vector database directory at {config.VECTOR_DB_DIR}")
            
    except Exception as e:
        logger.error(f"Error resetting database directory: {e}")
        raise RuntimeError(f"Failed to reset database: {e}")


def get_indexed_documents() -> List[Dict[str, Any]]:
    """
    Queries ChromaDB metadata to return a list of unique indexed filenames,
    along with their page counts and total chunks.
    """
    try:
        col = get_collection()
        results = col.get(include=["metadatas"])
        if not results or not results["metadatas"]:
            return []
        
        docs = {}
        for meta in results["metadatas"]:
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", 1)
            if source not in docs:
                docs[source] = {"pages": set(), "chunks": 0}
            docs[source]["pages"].add(page)
            docs[source]["chunks"] += 1
            
        return [
            {
                "filename": k,
                "pages": len(v["pages"]),
                "chunks": v["chunks"]
            } for k, v in docs.items()
        ]
    except Exception as e:
        logger.error(f"Failed to fetch indexed documents: {e}")
        return []


def delete_document_from_db(source_name: str) -> None:
    """
    Removes all chunks associated with a specific source document from vector storage.
    """
    try:
        col = get_collection()
        col.delete(where={"source": source_name})
        logger.info(f"Successfully deleted document '{source_name}' from vector DB.")
    except Exception as e:
        logger.error(f"Failed to delete document '{source_name}': {e}")
        raise RuntimeError(f"Deletion failed: {e}")

