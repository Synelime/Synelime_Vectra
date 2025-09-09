#!/usr/bin/env python3
"""
Complete Workflow Example

This script demonstrates how to use all three components (json_handler, data_cleaner, and vector_db_uploader)
together in a single workflow.
"""

import os
import json
import pandas as pd

# Import our custom modules
import json_handler
import data_cleaner
from vector_db_uploader import (
    DocumentEmbedder, 
    MockVectorDBUploader,
    process_and_upload_documents
)

def main():
    print("Starting complete workflow demonstration...\n")
    
    # Step 1: Use json_handler to load and process JSON data
    print("Step 1: Loading and processing JSON data")
    input_file = "example_documents.json"
    
    # Check if the example file exists, if not use messy_documents.json
    if not os.path.exists(input_file):
        input_file = "messy_documents.json"
        if not os.path.exists(input_file):
            print(f"Error: Neither example_documents.json nor messy_documents.json found.")
            return False
    
    # Load the JSON data
    documents = json_handler.read_json_file(input_file)
    print(f"Loaded {len(documents)} documents from {input_file}")
    
    # Extract relevant fields for processing
    extracted_data = []
    for doc in documents:
        extracted = {}
        # Extract common fields that should be present in most documents
        for field in ['id', 'title', 'content', 'author', 'date', 'tags']:
            if field in doc:
                extracted[field] = doc[field]
        extracted_data.append(extracted)
    
    # Save the extracted data to a temporary file
    temp_file = "temp_extracted_data.json"
    json_handler.write_json_file(extracted_data, temp_file)
    print(f"Extracted relevant fields and saved to {temp_file}\n")
    
    # Step 2: Use data_cleaner to clean and preprocess the data
    print("Step 2: Cleaning and preprocessing data")
    # Convert JSON to DataFrame for cleaning
    df = pd.DataFrame(extracted_data)
    
    # Apply data cleaning operations
    cleaned_df = data_cleaner.remove_duplicates(df, subset=['id'] if 'id' in df.columns else None)
    cleaned_df = data_cleaner.handle_missing_values(cleaned_df)
    
    # Clean text data if content column exists
    if 'content' in cleaned_df.columns:
        cleaned_df = data_cleaner.clean_text_data(cleaned_df, ['content'])
    
    # Extract text features if content column exists
    if 'content' in cleaned_df.columns:
        cleaned_df = data_cleaner.extract_features(cleaned_df, 'content')
    
    # Save the cleaned data
    processed_file = "workflow_processed_data.json"
    data_cleaner.save_processed_data(cleaned_df, processed_file)
    print(f"Cleaned data and saved to {processed_file}\n")
    
    # Step 3: Use vector_db_uploader to generate embeddings and upload to a vector database
    print("Step 3: Generating embeddings and uploading to vector database")
    
    # Initialize document embedder
    embedder = DocumentEmbedder()
    
    # Initialize mock uploader (for demonstration)
    uploader = MockVectorDBUploader(embedder, output_dir="./mock_vector_db")
    
    # Process and upload documents
    success = process_and_upload_documents(
        input_path=processed_file,
        collection_name="documents",
        uploader=uploader,
        text_field="content",
        id_field="id" if 'id' in cleaned_df.columns else None,
        clean_data=False,  # We've already cleaned the data
        metadata_fields=[col for col in cleaned_df.columns if col not in ['id', 'content']]
    )
    
    if success:
        print("\nComplete workflow demonstration finished successfully!")
        print("\nSummary:")
        print(f"1. Loaded and processed documents from {input_file}")
        print(f"2. Cleaned and preprocessed data, saved to {processed_file}")
        print(f"3. Generated embeddings and uploaded to mock vector database")
        return True
    else:
        print("\nWorkflow demonstration failed at the vector database upload step.")
        return False

if __name__ == "__main__":
    main()