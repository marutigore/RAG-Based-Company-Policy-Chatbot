"""
Automated Test Verification Module.
Implements 5 tests: imports, config, chunking, embedding, and end-to-end pipeline.
"""

import os
import unittest
import logging
from unittest.mock import patch, MagicMock

# Setup logger
logger = logging.getLogger("test_validation")
logging.basicConfig(level=logging.INFO)


class TestValidationSuite(unittest.TestCase):
    """
    Validation tests for Project 1: Multi-Document RAG Company Policy Chatbot.
    """

    def test_01_imports(self) -> None:
        """
        Test 1: Verification of core and custom imports.
        Ensures all dependencies and custom utilities are properly installed and visible.
        """
        logger.info("Running Test 1: Imports Verification...")
        try:
            # Import third-party dependencies
            import streamlit
            import fitz  # PyMuPDF
            import chromadb
            import openai
            import tiktoken
            import dotenv
            
            # Import project modules
            import config
            from utils import document_loader, chunker, embedder, retriever, validator
            
            logger.info("Test 1 passed: All imports resolved successfully.")
        except ImportError as e:
            logger.error(f"Test 1 failed: Import resolution error: {e}")
            self.fail(f"Failed to import necessary module: {e}")

    def test_02_config(self) -> None:
        """
        Test 2: Verification of configuration loading.
        Verifies default constants exist and check_keys behaves predictably.
        """
        logger.info("Running Test 2: Configuration Verification...")
        import config
        
        # Verify default parameters exist
        self.assertIsNotNone(config.EMBEDDING_MODEL)
        self.assertIsNotNone(config.LLM_MODEL)
        self.assertTrue(isinstance(config.CHUNK_SIZE, int))
        self.assertTrue(isinstance(config.CHUNK_OVERLAP, int))
        self.assertIsNotNone(config.VECTOR_DB_DIR)
        
        # Check API key validation logic
        key_validity = config.check_keys()
        self.assertIn(key_validity, [True, False])
        logger.info(f"Test 2 passed: Configuration is valid. API Key Present: {key_validity}")

    def test_03_chunking(self) -> None:
        """
        Test 3: Verification of chunking and token counting.
        Ensures split_documents creates correct chunk structures and splits cleanly.
        """
        logger.info("Running Test 3: Chunking Logic Verification...")
        from utils.chunker import split_documents, count_tokens
        
        sample_pages = [
            {
                "text": "This is page one. It contains sample data. We are checking how RAG works.",
                "metadata": {"source": "test_policy.pdf", "page": 1}
            },
            {
                "text": "This is page two. This details guidelines, employee benefits, and leaves.",
                "metadata": {"source": "test_policy.pdf", "page": 2}
            }
        ]
        
        # Run chunking with small size to guarantee splits
        chunks = split_documents(sample_pages, chunk_size=10, chunk_overlap=2)
        
        # Basic validation on results list
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)
            self.assertEqual(chunk["metadata"]["source"], "test_policy.pdf")
            self.assertTrue("chunk_idx" in chunk["metadata"])
            self.assertTrue(chunk["metadata"]["token_count"] > 0)
            
        logger.info(f"Test 3 passed: Chunking generated {len(chunks)} text chunks successfully.")

    @patch("utils.embedder._get_model")
    def test_04_embedding(self, mock_get_model: MagicMock) -> None:
        """
        Test 4: Verification of embedding routines.
        Mocks the local SentenceTransformer model call to run verification locally.
        """
        logger.info("Running Test 4: Embedding Logic Verification...")
        from utils.embedder import get_embedding
        
        # Mocking the SentenceTransformer model encode method
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384  # Return a mock 384 vector
        mock_get_model.return_value = mock_model
        
        try:
            vector = get_embedding("Testing embeddings generation")
            self.assertEqual(len(vector), 1536)
            self.assertEqual(vector[0], 0.1)
            logger.info("Test 4 passed: Embedder returned correct dimensions (1536 floats) using local mocked model.")
        finally:
            pass

    @patch("utils.retriever.get_collection")
    @patch("openai.OpenAI")
    def test_05_end_to_end_pipeline(
        self,
        mock_openai: MagicMock,
        mock_collection: MagicMock
    ) -> None:
        """
        Test 5: Integration End-to-End Pipeline Verification.
        Checks mock components working sequentially from query to final answer output.
        """
        logger.info("Running Test 5: End-to-End Pipeline Verification...")
        
        # 1. Mock DB collection query response
        mock_query_results = {
            "documents": [["Retrieved company policy chunk text about vacations and sick leave."]],
            "metadatas": [[{"source": "policy.pdf", "page": 3}]],
            "distances": [[0.1]]
        }
        mock_collection.return_value.query.return_value = mock_query_results
        
        # 2. Mock Embedder response
        mock_embed_data = MagicMock()
        mock_embed_data.embedding = [0.0] * 1536
        mock_embed_res = MagicMock()
        mock_embed_res.data = [mock_embed_data]
        
        # 3. Mock App LLM response
        mock_choice = MagicMock()
        mock_choice.message.content = "According to Section 3, employees get 15 days of vacation."
        mock_llm_res = MagicMock()
        mock_llm_res.choices = [mock_choice]
        
        # 4. Mock Validator LLM Judge response (Faithfulness & Relevancy)
        mock_faith_choice = MagicMock()
        mock_faith_choice.message.content = '{"score": 1.0, "reasoning": "Grounded in retrieved context."}'
        mock_faith_res = MagicMock()
        mock_faith_res.choices = [mock_faith_choice]
        
        mock_rel_choice = MagicMock()
        mock_rel_choice.message.content = '{"score": 0.95, "reasoning": "Directly answers the question."}'
        mock_rel_res = MagicMock()
        mock_rel_res.choices = [mock_rel_choice]
        
        # Configure mock OpenAI client behaviors
        client_mock = mock_openai.return_value
        client_mock.embeddings.create.return_value = mock_embed_res
        client_mock.chat.completions.create.side_effect = [
            mock_llm_res,      # Called by app.py to get answer
            mock_faith_res,    # Called by validator.py for faithfulness
            mock_rel_res       # Called by validator.py for relevancy
        ]

        # Temporarily force API key configured for RAG flow
        import config
        original_key = config.OPENAI_API_KEY
        config.OPENAI_API_KEY = "mock-key-for-testing"

        try:
            import app
            res = app.run_pipeline("What is the vacation policy?")
            
            # Verify results
            self.assertIn("answer", res)
            self.assertIn("citations", res)
            self.assertIn("evaluation", res)
            
            self.assertEqual(res["answer"], "According to Section 3, employees get 15 days of vacation.")
            self.assertEqual(len(res["citations"]), 1)
            self.assertEqual(res["citations"][0]["metadata"]["source"], "policy.pdf")
            self.assertEqual(res["evaluation"]["faithfulness"]["score"], 1.0)
            self.assertEqual(res["evaluation"]["relevancy"]["score"], 0.95)
            
            logger.info("Test 5 passed: Mocked end-to-end RAG pipeline completed successfully.")
        finally:
            config.OPENAI_API_KEY = original_key


if __name__ == "__main__":
    unittest.main()
