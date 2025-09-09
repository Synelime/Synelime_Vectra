#!/usr/bin/env python3
"""
Test Pinecone Integration

This script tests the Pinecone integration with the updated vector_db_uploader.py code.
You'll need to provide your own Pinecone API key to run this script.
"""

import os
import json
from vector_db_uploader import DocumentEmbedder, PineconeUploader, process_and_upload_documents

def main():
    # Check if Pinecone API key is provided
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set.")
        print("Please set your Pinecone API key as an environment variable:")
        print("  On Windows: set PINECONE_API_KEY=your_api_key")
        print("  On macOS/Linux: export PINECONE_API_KEY=your_api_key")
        return False
    
    # Use example_documents.json if it exists, otherwise use a sample document
    input_file = "example_documents.json"
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found. Creating a sample document.")
        sample_data = [
            {
                "id": "doc1",
                "title": "Test Document",
                "content": "This is a test document for Pinecone integration.",
                "author": "Test User",
                "tags": ["test", "pinecone", "integration"]
            }
        ]
        with open("sample_pinecone_test.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        input_file = "sample_pinecone_test.json"
        print(f"Created sample file: {input_file}")
    
    # Initialize document embedder
    print("Initializing document embedder...")
    embedder = DocumentEmbedder()
    
    # Initialize Pinecone uploader
    print("Initializing Pinecone uploader...")
    index_name = "test-documents"
    uploader = PineconeUploader(embedder, api_key=api_key)
    
    # Process and upload documents
    print(f"Processing and uploading documents from {input_file}...")
    success = process_and_upload_documents(
        input_path=input_file,
        collection_name=index_name,
        uploader=uploader,
        text_field="content",
        id_field="id",
        clean_data=True,
        metadata_fields=["title", "author", "tags"]
    )
    
    if success:
        print("\nPinecone integration test completed successfully!")
        return True
    else:
        print("\nPinecone integration test failed.")
        return False

if __name__ == "__main__":
    main()