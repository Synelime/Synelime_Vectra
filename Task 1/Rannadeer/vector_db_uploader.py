#!/usr/bin/env python3
"""
Vector Database Document Uploader

This script provides utilities for uploading documents to vector databases.
It supports document embedding generation and uploading to different vector database backends.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Callable
from pathlib import Path

# Import local modules
import json_handler
import data_cleaner

# Note: The actual vector database libraries would need to be installed
# For example: pip install sentence-transformers chromadb pinecone-client

# For demonstration purposes, we'll include import statements but make the actual
# functionality modular so it can work with or without these libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using mock embeddings.")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. ChromaDB functionality will be limited.")

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except (ImportError, Exception) as e:
    PINECONE_AVAILABLE = False
    print(f"Warning: Pinecone not available. Error: {e}")
    print("Pinecone functionality will be limited.")


class DocumentEmbedder:
    """
    Class for generating embeddings from document text.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
        else:
            print("Using mock embeddings as sentence-transformers is not available")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is not None:
            return self.model.encode(text)
        else:
            # Mock embedding for demonstration (random vector)
            # In a real scenario, you would need the actual embedding model
            return np.random.rand(384)  # Common embedding size
    
    def batch_generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            Array of embedding vectors
        """
        if self.model is not None:
            return self.model.encode(texts)
        else:
            # Mock embeddings for demonstration
            return np.random.rand(len(texts), 384)


class VectorDBUploader:
    """
    Base class for vector database uploaders.
    """
    def __init__(self, embedder: DocumentEmbedder):
        """
        Initialize the uploader with an embedder.
        
        Args:
            embedder: DocumentEmbedder instance for generating embeddings
        """
        self.embedder = embedder
    
    def prepare_documents(self, documents: List[Dict], text_field: str) -> List[Dict]:
        """
        Prepare documents for uploading by adding embeddings.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to embed
            
        Returns:
            Documents with added embedding vectors
        """
        prepared_docs = []
        
        for doc in documents:
            if text_field in doc and doc[text_field]:
                doc_copy = doc.copy()
                text = doc_copy[text_field]
                
                # Generate embedding
                embedding = self.embedder.generate_embedding(text)
                doc_copy['embedding'] = embedding.tolist()  # Convert numpy array to list for JSON serialization
                
                prepared_docs.append(doc_copy)
            else:
                print(f"Warning: Document missing '{text_field}' field or empty value. Skipping.")
        
        return prepared_docs
    
    def upload_documents(self, documents: List[Dict], collection_name: str, **kwargs) -> bool:
        """
        Upload documents to the vector database.
        To be implemented by subclasses.
        
        Args:
            documents: List of prepared document dictionaries with embeddings
            collection_name: Name of the collection to upload to
            **kwargs: Additional arguments specific to the database backend
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement upload_documents method")


class ChromaDBUploader(VectorDBUploader):
    """
    Uploader for ChromaDB vector database.
    """
    def __init__(self, embedder: DocumentEmbedder, persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB uploader.
        
        Args:
            embedder: DocumentEmbedder instance
            persist_directory: Directory to persist the ChromaDB database
        """
        super().__init__(embedder)
        self.persist_directory = persist_directory
        self.client = None
        
        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=persist_directory)
                print(f"Connected to ChromaDB at {persist_directory}")
            except Exception as e:
                print(f"Error connecting to ChromaDB: {e}")
        else:
            print("ChromaDB functionality limited as the library is not installed")
    
    def upload_documents(self, documents: List[Dict], collection_name: str, 
                         id_field: str = "id", text_field: str = "content", 
                         metadata_fields: List[str] = None, **kwargs) -> bool:
        """
        Upload documents to ChromaDB.
        
        Args:
            documents: List of document dictionaries
            collection_name: Name of the ChromaDB collection
            id_field: Field to use as document ID
            text_field: Field containing the text to embed
            metadata_fields: Fields to include as metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not CHROMADB_AVAILABLE or self.client is None:
            print("ChromaDB not available. Cannot upload documents.")
            return False
        
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # Prepare data for ChromaDB format
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []
            
            for doc in documents:
                # Skip if missing required fields
                if id_field not in doc or text_field not in doc:
                    print(f"Warning: Document missing required fields. Skipping.")
                    continue
                
                # Extract ID and text
                doc_id = str(doc[id_field])  # Ensure ID is string
                text = doc[text_field]
                
                # Generate embedding if not already present
                if 'embedding' not in doc:
                    embedding = self.embedder.generate_embedding(text)
                else:
                    embedding = np.array(doc['embedding'])
                
                # Extract metadata
                metadata = {}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in doc and field != id_field and field != text_field:
                            metadata[field] = doc[field]
                
                ids.append(doc_id)
                embeddings.append(embedding)
                metadatas.append(metadata)
                documents_text.append(text)
            
            # Add documents to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            print(f"Successfully uploaded {len(ids)} documents to ChromaDB collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error uploading to ChromaDB: {e}")
            return False


class PineconeUploader(VectorDBUploader):
    """
    Uploader for Pinecone vector database.
    """
    def __init__(self, embedder: DocumentEmbedder, api_key: str, environment: str = None):
        """
        Initialize the Pinecone uploader.
        
        Args:
            embedder: DocumentEmbedder instance
            api_key: Pinecone API key
            environment: Pinecone environment (deprecated, kept for backward compatibility)
        """
        super().__init__(embedder)
        self.api_key = api_key
        self.environment = environment
        self.initialized = False
        
        if PINECONE_AVAILABLE:
            try:
                # Initialize with new Pinecone client
                self.pc = Pinecone(api_key=api_key)
                self.initialized = True
                print("Connected to Pinecone")
            except Exception as e:
                print(f"Error connecting to Pinecone: {e}")
        else:
            print("Pinecone functionality limited as the library is not installed")
    
    def upload_documents(self, documents: List[Dict], collection_name: str, 
                         id_field: str = "id", text_field: str = "content", 
                         metadata_fields: List[str] = None, 
                         dimension: int = 384, **kwargs) -> bool:
        """
        Upload documents to Pinecone.
        
        Args:
            documents: List of document dictionaries
            collection_name: Name of the Pinecone index
            id_field: Field to use as document ID
            text_field: Field containing the text to embed
            metadata_fields: Fields to include as metadata
            dimension: Dimension of the embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if not PINECONE_AVAILABLE or not self.initialized:
            print("Pinecone not available. Cannot upload documents.")
            return False
        
        try:
            # Check if index exists, create if not
            if collection_name not in [index.name for index in self.pc.list_indexes()]:
                # Create index with serverless spec
                self.pc.create_index(
                    name=collection_name,
                    dimension=dimension,
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )
                print(f"Created new Pinecone index: {collection_name}")
            
            # Connect to index
            index = self.pc.Index(collection_name)
            
            # Prepare data for Pinecone format
            vectors_to_upsert = []
            
            for doc in documents:
                # Skip if missing required fields
                if id_field not in doc or text_field not in doc:
                    print(f"Warning: Document missing required fields. Skipping.")
                    continue
                
                # Extract ID and text
                doc_id = str(doc[id_field])  # Ensure ID is string
                text = doc[text_field]
                
                # Generate embedding if not already present
                if 'embedding' not in doc:
                    embedding = self.embedder.generate_embedding(text)
                else:
                    embedding = doc['embedding']
                
                # Extract metadata
                metadata = {}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in doc and field != id_field:
                            metadata[field] = doc[field]
                
                # Always include the text in metadata for retrieval
                metadata[text_field] = text
                
                vectors_to_upsert.append({
                    'id': doc_id,
                    'values': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    'metadata': metadata
                })
            
            # Upsert in batches (Pinecone has limits on batch size)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                index.upsert(vectors=batch)
            
            print(f"Successfully uploaded {len(vectors_to_upsert)} documents to Pinecone index '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error uploading to Pinecone: {e}")
            return False


class MockVectorDBUploader(VectorDBUploader):
    """
    Mock uploader for demonstration and testing without actual vector database dependencies.
    """
    def __init__(self, embedder: DocumentEmbedder, output_dir: str = "./mock_vector_db"):
        """
        Initialize the mock uploader.
        
        Args:
            embedder: DocumentEmbedder instance
            output_dir: Directory to save the mock database files
        """
        super().__init__(embedder)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def upload_documents(self, documents: List[Dict], collection_name: str, 
                         id_field: str = "id", text_field: str = "content", **kwargs) -> bool:
        """
        Save documents to a JSON file as a mock vector database.
        
        Args:
            documents: List of document dictionaries
            collection_name: Name of the collection (used for filename)
            id_field: Field to use as document ID
            text_field: Field containing the text to embed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare documents with embeddings if not already present
            prepared_docs = []
            
            for doc in documents:
                doc_copy = doc.copy()
                
                # Generate embedding if not already present and text field exists
                if 'embedding' not in doc and text_field in doc and doc[text_field]:
                    embedding = self.embedder.generate_embedding(doc[text_field])
                    doc_copy['embedding'] = embedding.tolist()  # Convert numpy array to list for JSON serialization
                
                prepared_docs.append(doc_copy)
            
            # Save to JSON file
            output_path = os.path.join(self.output_dir, f"{collection_name}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(prepared_docs, f, indent=2)
            
            print(f"Successfully saved {len(prepared_docs)} documents to mock database: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving to mock database: {e}")
            return False


def process_and_upload_documents(input_path: str, collection_name: str, 
                                uploader: VectorDBUploader,
                                text_field: str = "content",
                                id_field: str = "id",
                                clean_data: bool = True,
                                metadata_fields: List[str] = None) -> bool:
    """
    Process documents from a file and upload them to a vector database.
    
    Args:
        input_path: Path to the input file (JSON, CSV, etc.)
        collection_name: Name of the collection/index to upload to
        uploader: VectorDBUploader instance
        text_field: Field containing the text to embed
        id_field: Field to use as document ID
        clean_data: Whether to clean the data before uploading
        metadata_fields: Fields to include as metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load data into DataFrame
        df = data_cleaner.load_data_to_dataframe(input_path)
        
        # Clean data if requested
        if clean_data:
            # Normalize column names
            df = data_cleaner.normalize_column_names(df)
            
            # Update field names if they were normalized
            text_field = text_field.lower().replace(' ', '_')
            id_field = id_field.lower().replace(' ', '_')
            if metadata_fields:
                metadata_fields = [field.lower().replace(' ', '_') for field in metadata_fields]
            
            # Remove duplicates
            df = data_cleaner.remove_duplicates(df)
            
            # Clean text data
            if text_field in df.columns:
                df = data_cleaner.clean_text_data(df, text_columns=[text_field])
            
            # Handle missing values
            df = data_cleaner.handle_missing_values(df, strategy='drop')
        
        # Ensure ID field exists
        if id_field not in df.columns:
            print(f"Warning: ID field '{id_field}' not found. Creating sequential IDs.")
            df[id_field] = [f"doc_{i}" for i in range(len(df))]
        
        # Convert DataFrame to list of dictionaries
        documents = df.to_dict(orient='records')
        
        # Upload to vector database
        success = uploader.upload_documents(
            documents=documents,
            collection_name=collection_name,
            id_field=id_field,
            text_field=text_field,
            metadata_fields=metadata_fields
        )
        
        return success
        
    except Exception as e:
        print(f"Error processing and uploading documents: {e}")
        return False


def main():
    """
    Example usage of the document upload functionality.
    """
    # Create a sample JSON file if it doesn't exist
    sample_file = "sample_documents.json"
    if not os.path.exists(sample_file):
        sample_data = [
            {
                "id": "doc1",
                "title": "Introduction to Vector Databases",
                "content": "Vector databases are specialized database systems designed to store and query vector embeddings efficiently.",
                "author": "Ali",
                "tags": ["vector", "database", "embeddings"]
            },
            {
                "id": "doc2",
                "title": "Working with JSON in Python",
                "content": "JSON (JavaScript Object Notation) is a lightweight data interchange format that is easy for humans to read and write.",
                "author": "Ali",
                "tags": ["json", "python", "data"]
            },
            {
                "id": "doc3",
                "title": "Data Cleaning Techniques",
                "content": "Data cleaning is the process of fixing or removing incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data within a dataset.",
                "author": "Ali",
                "tags": ["data", "cleaning", "preprocessing"]
            }
        ]
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Created sample file: {sample_file}")
    
    # Initialize document embedder
    embedder = DocumentEmbedder()
    
    # Initialize mock uploader (for demonstration)
    uploader = MockVectorDBUploader(embedder, output_dir="./mock_vector_db")
    
    # Process and upload documents
    success = process_and_upload_documents(
        input_path=sample_file,
        collection_name="documents",
        uploader=uploader,
        text_field="content",
        id_field="id",
        clean_data=True,
        metadata_fields=["title", "author", "tags"]
    )
    
    if success:
        print("Document upload process completed successfully!")
    else:
        print("Document upload process failed.")


if __name__ == "__main__":
    main()