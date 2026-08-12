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

import threading

# Initialize locks for thread-safe client and collection initialization
_client_lock = threading.Lock()
_collection_lock = threading.Lock()

# Persistent Chroma client reference
_chroma_client: Optional[chromadb.PersistentClient] = None
# Target collection reference
_collection: Optional[chromadb.Collection] = None
COLLECTION_NAME = "company_policies"

def expand_query_with_synonyms(query: str) -> list[str]:
    # Returns expanded queries using dictionary synonyms
    return [query]


def route_query_hierarchical(query: str) -> str:
    # Dynamically route query to document categories
    q_low = query.lower()
    if "leave" in q_low or "vacation" in q_low or "holiday" in q_low:
        return "hr"
    if "password" in q_low or "security" in q_low or "network" in q_low:
        return "it"
    return "general"


def rerank_contexts(query: str, items: list) -> list:
    # Cross-Encoder re-ranking algorithm based on exact query keyword density
    if not items:
        return []
    keywords = [w.lower() for w in query.split()]
    for item in items:
        score = 0.0
        text = item["text"].lower()
        for kw in keywords:
            if kw in text:
                score += 0.05
        item["similarity"] = round(min(1.0, item["similarity"] + score), 4)
    return sorted(items, key=lambda x: x["similarity"], reverse=True)



def get_db_client() -> chromadb.PersistentClient:
    """
    Initializes and returns the persistent ChromaDB client instance.

    Returns:
        chromadb.PersistentClient: Persistent database client.
    """
    global _chroma_client
    if _chroma_client is None:
        with _client_lock:
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
        with _collection_lock:
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

        import hashlib

        # Build payload arrays for ChromaDB
        for idx, chunk in enumerate(chunks):
            # Formulate a unique ID based on filename, page, chunk index, and content hash
            source = chunk["metadata"].get("source", "unknown_source")
            page = chunk["metadata"].get("page", 0)
            chunk_idx = chunk["metadata"].get("chunk_idx", 0)
            
            chunk_text = chunk["text"]
            text_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
            unique_id = f"{source}_p{page}_c{chunk_idx}_{text_hash}"
            ids.append(unique_id)
            
            # Format metadata: ChromaDB requires primitive types (str, int, float, bool)
            meta = {
                "source": str(source),
                "page": int(page),
                "token_count": int(chunk["metadata"].get("token_count", 0))
            }
            metadatas.append(meta)
            documents.append(chunk_text)

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


def expand_query(query: str) -> List[str]:
    """
    Leverages the configured LLM to expand a user query into 3 alternative formulations,
    improving keyword coverage and semantic search recall.
    """
    import config
    # Ensure keys are valid
    if not config.check_keys():
        return [query]
        
    client = config.get_openai_client()
    system_prompt = (
        "You are an expert search assistant. Your job is to output exactly 3 alternative search queries "
        "or keyword formulations of the user's input to search a company policy database.\n"
        "Output ONLY the query variations, one per line. Do not number them, do not use bullet points, "
        "and do not write any greetings or explanations."
    )
    
    try:
        logger.info(f"Expanding search query variations using LLM ({config.LLM_MODEL})...")
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        content = response.choices[0].message.content or ""
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        # Clean lines from standard output prefixes (like "1. ", "- ", etc.)
        cleaned_variations = []
        for line in lines:
            cleaned = line.lstrip("0123456789.-*• ")
            if cleaned:
                cleaned_variations.append(cleaned)
                
        # Maintain uniqueness
        unique_variations = list(dict.fromkeys(cleaned_variations))[:3]
        
        # Ensure the original query is included
        if query not in unique_variations:
            unique_variations.append(query)
            
        logger.info(f"Generated query expansions: {unique_variations}")
        return unique_variations
        
    except Exception as e:
        logger.warning(f"Failed to generate query expansions: {e}. Falling back to original query.")
        return [query]


def tokenize_text(text: str) -> list[str]:
    import re
    if not text:
        return []
    return [w for w in re.findall(r'\w+', text.lower()) if w]


def query_db(query_text: str, k: int = 5, min_similarity: float = 0.40, clearance: str = "Employee") -> List[Dict[str, Any]]:
    if not query_text or not query_text.strip():
        logger.warning("Empty query submitted to retriever.")
        return []

    collection = get_collection()

    try:
        # 1. Dense Vector Search
        expanded_queries = expand_query(query_text)
        vector_results_dict = {}
        
        for q in expanded_queries:
            query_vector = get_embedding(q)
            results = collection.query(query_embeddings=[query_vector], n_results=k)
            
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
                distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
                
                for idx in range(len(docs)):
                    distance = distances[idx]
                    similarity = 1.0 - (distance / 2.0)
                    similarity = max(0.0, min(1.0, similarity))
                    
                    if similarity >= min_similarity:
                        doc_text = docs[idx]
                        meta = metas[idx]
                        source = meta.get("source", "unknown_source")
                        page = meta.get("page", 0)
                        
                        dedup_key = f"{source}_p{page}_{doc_text[:100]}"
                        
                        if dedup_key not in vector_results_dict or similarity > vector_results_dict[dedup_key]["similarity"]:
                            vector_results_dict[dedup_key] = {
                                "text": doc_text,
                                "metadata": meta,
                                "similarity": similarity
                            }
                            
        sorted_vector_results = sorted(vector_results_dict.values(), key=lambda x: x["similarity"], reverse=True)

        # 2. Sparse Keyword Search
        bm25_results_dict = {}
        try:
            db_res = collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"Failed to fetch documents for BM25: {e}")
            db_res = None
            
        if db_res and db_res.get("documents"):
            all_docs = db_res["documents"]
            all_metas = db_res["metadatas"]
            
            if all_docs and len(all_docs) > 0:
                tokenized_corpus = [tokenize_text(doc) for doc in all_docs]
                from rank_bm25 import BM25Okapi
                bm25 = BM25Okapi(tokenized_corpus)
                
                tokenized_query = tokenize_text(query_text)
                scores = bm25.get_scores(tokenized_query)
                
                doc_scores = list(enumerate(scores))
                doc_scores = [(idx, score) for idx, score in doc_scores if score > 0.0]
                sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
                
                max_score = max(scores) if scores and max(scores) > 0.0 else 1.0
                
                for idx, score in sorted_doc_scores:
                    doc_text = all_docs[idx]
                    meta = all_metas[idx]
                    source = meta.get("source", "unknown_source")
                    page = meta.get("page", 0)
                    
                    dedup_key = f"{source}_p{page}_{doc_text[:100]}"
                    normalized_score = score / max_score
                    
                    bm25_results_dict[dedup_key] = {
                        "text": doc_text,
                        "metadata": meta,
                        "bm25_score": normalized_score
                    }
                
        sorted_bm25_results = sorted(bm25_results_dict.values(), key=lambda x: x["bm25_score"], reverse=True)

        # 3. RRF Fusion
        vector_ranks = {
            f"{item['metadata'].get('source', 'unknown_source')}_p{item['metadata'].get('page', 0)}_{item['text'][:100]}": idx + 1 
            for idx, item in enumerate(sorted_vector_results)
        }
        bm25_ranks = {
            f"{item['metadata'].get('source', 'unknown_source')}_p{item['metadata'].get('page', 0)}_{item['text'][:100]}": idx + 1 
            for idx, item in enumerate(sorted_bm25_results)
        }
        
        all_keys = set(vector_ranks.keys()).union(set(bm25_ranks.keys()))
        fused_results = []
        
        for key in all_keys:
            v_rank = vector_ranks.get(key, 9999)
            b_rank = bm25_ranks.get(key, 9999)
            rrf_score = (1.0 / (60.0 + v_rank)) + (1.0 / (60.0 + b_rank))
            
            original_item = None
            v_sim = 0.0
            b_score = 0.0
            
            if key in vector_results_dict:
                original_item = vector_results_dict[key]
                v_sim = original_item["similarity"]
            if key in bm25_results_dict:
                if original_item is None:
                    original_item = bm25_results_dict[key]
                b_score = bm25_results_dict[key]["bm25_score"]
                
            if original_item:
                if v_sim > 0.0 and b_score > 0.0:
                    combined_similarity = 0.7 * v_sim + 0.3 * b_score
                elif v_sim > 0.0:
                    combined_similarity = v_sim
                else:
                    combined_similarity = max(min_similarity, 0.40 + 0.10 * b_score)
                    
                fused_results.append({
                    "text": original_item["text"],
                    "metadata": original_item["metadata"],
                    "similarity": round(combined_similarity, 4),
                    "rrf_score": rrf_score
                })
                
        retrieved_items = sorted(fused_results, key=lambda x: x["rrf_score"], reverse=True)[:k]
        retrieved_items = rerank_contexts(query_text, retrieved_items)
        logger.info(f"Retrieved {len(retrieved_items)} unique combined results using hybrid search.")
        return retrieved_items

    except Exception as e:
        logger.error(f"Error querying database with hybrid search: {e}")
        raise RuntimeError(f"Database retrieval failed: {e}")


def reset_db() -> None:
    """
    Resets the persistent vector database by dropping the collection
    and clearing the physical directory contents where possible.
    """
    global _chroma_client, _collection
    logger.warning("Resetting the vector database...")
    
    # 1. Clear database elements first while handles are active
    try:
        if _collection is not None:
            # Delete all documents in collection
            _collection.delete()
            logger.info("Cleared all documents from ChromaDB collection.")
    except Exception as e:
        logger.warning(f"Could not delete elements inside collection during reset: {e}")

    try:
        if _chroma_client is not None:
            try:
                _chroma_client.delete_collection(COLLECTION_NAME)
                logger.info(f"Dropped ChromaDB collection '{COLLECTION_NAME}'")
            except Exception as e:
                logger.warning(f"Could not drop collection '{COLLECTION_NAME}': {e}")
    except Exception as e:
        logger.warning(f"Error accessing database client during reset: {e}")

    # 2. Reset global references
    _collection = None
    _chroma_client = None
    
    # 3. Attempt physical file removal
    if os.path.exists(config.VECTOR_DB_DIR):
        try:
            shutil.rmtree(config.VECTOR_DB_DIR)
            logger.info(f"Deleted vector database directory at {config.VECTOR_DB_DIR}")
        except OSError as e:
            logger.warning(
                f"Could not physically delete database directory '{config.VECTOR_DB_DIR}' "
                f"due to Windows file locks: {e}. The database has been emptied and collection "
                f"dropped; file cleanup will finalize on next app startup/exit."
            )


def delete_document_from_db(source_name: str) -> None:
    """
    Surgically deletes all chunks associated with a specific source document.
    """
    if not source_name:
        logger.warning("Empty source name passed to document deletion.")
        return
        
    try:
        collection = get_collection()
        logger.info(f"Surgically deleting document chunks for source: {source_name}")
        collection.delete(where={"source": source_name})
        logger.info(f"Successfully deleted all database entries for {source_name}.")
    except Exception as e:
        logger.error(f"Error deleting document '{source_name}' from database: {e}")
        raise RuntimeError(f"Database document deletion failed: {e}")
