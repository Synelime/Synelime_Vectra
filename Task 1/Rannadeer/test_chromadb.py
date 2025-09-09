#!/usr/bin/env python3
"""
Test ChromaDB Integration

This script tests the ChromaDB integration with the vector_db_uploader.py code.
"""

import os
import json
from vector_db_uploader import DocumentEmbedder, ChromaDBUploader, process_and_upload_documents

def main():
    # Use example_documents.json if it exists, otherwise use a sample document
    input_file = "example_documents.json"
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found. Creating a sample document.")
        sample_data = [
            {
                "id": "doc1",
                "title": "Test Document",
                "content": "This is a test document for ChromaDB integration.",
                "author": "Test User",
                "tags": ["test", "chromadb", "integration"]
            }
        ]
        with open("sample_chromadb_test.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        input_file = "sample_chromadb_test.json"
        print(f"Created sample file: {input_file}")
    
    # Initialize document embedder
    print("Initializing document embedder...")
    embedder = DocumentEmbedder()
    
    # Initialize ChromaDB uploader
    print("Initializing ChromaDB uploader...")
    persist_directory = "./chroma_db"
    uploader = ChromaDBUploader(embedder, persist_directory=persist_directory)
    
    # Process and upload documents
    print(f"Processing and uploading documents from {input_file}...")
    success = process_and_upload_documents(
        input_path=input_file,
        collection_name="test_documents",
        uploader=uploader,
        text_field="content",
        id_field="id",
        clean_data=True,
        metadata_fields=["title", "author", "tags"]
    )
    
    if success:
        print("\nChromaDB integration test completed successfully!")
        print(f"ChromaDB data stored in: {os.path.abspath(persist_directory)}")
        return True
    else:
        print("\nChromaDB integration test failed.")
        return False

if __name__ == "__main__":
    main()