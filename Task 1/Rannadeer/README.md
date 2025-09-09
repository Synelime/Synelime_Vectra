# Document Upload Scripts for Vector Database

This repository contains a collection of Python scripts for handling, cleaning, and uploading documents to vector databases. These scripts are designed to help you process document data and prepare it for advanced search capabilities using vector embeddings.

## Overview

The repository includes the following components:

1. **JSON Handler** (`json_handler.py`): Utilities for working with JSON files, including reading, writing, and manipulating JSON data.

2. **Data Cleaner** (`data_cleaner.py`): Tools for cleaning and preprocessing document data using Pandas and NumPy.

3. **Vector Database Uploader** (`vector_db_uploader.py`): Scripts for generating document embeddings and uploading them to vector databases.

4. **Example Data** (`example_documents.json` and `messy_documents.json`): Sample JSON files for testing the scripts.

## Prerequisites

Before using these scripts, you need to install the required Python packages:

```bash
pip install pandas numpy
```

For the full functionality of the vector database uploader, you may also need to install:

```bash
pip install sentence-transformers chromadb pinecone-client
```

Note: The scripts are designed to work even if these optional packages are not installed, but with limited functionality.

## Usage Guide

### 1. JSON Handler

The `json_handler.py` script provides utilities for working with JSON files.

**Basic Usage:**

```python
import json_handler

# Read a JSON file
data = json_handler.read_json_file("example_documents.json")

# Write data to a JSON file
json_handler.write_json_file(data, "output.json", pretty=True)

# Extract specific fields
subset = json_handler.extract_fields(data[0], ["title", "content"])

# Flatten a nested JSON structure
flattened = json_handler.flatten_json(data[0])
```

**Running the Example:**

```bash
python json_handler.py
```

### 2. Data Cleaner

The `data_cleaner.py` script provides tools for cleaning and preprocessing document data.

**Basic Usage:**

```python
import data_cleaner

# Load data into a DataFrame
df = data_cleaner.load_data_to_dataframe("messy_documents.json")

# Normalize column names
df = data_cleaner.normalize_column_names(df)

# Remove duplicates
df = data_cleaner.remove_duplicates(df)

# Clean text data
df = data_cleaner.clean_text_data(df, text_columns=["content"])

# Handle missing values
df = data_cleaner.handle_missing_values(df, strategy="fill", fill_values={"content": "No content available"})

# Save processed data
data_cleaner.save_processed_data(df, "cleaned_documents.json", format="json")
```

**Running the Example:**

```bash
python data_cleaner.py
```

### 3. Vector Database Uploader

The `vector_db_uploader.py` script provides tools for generating document embeddings and uploading them to vector databases.

**Basic Usage:**

```python
import vector_db_uploader

# Initialize document embedder
embedder = vector_db_uploader.DocumentEmbedder()

# Initialize a vector database uploader
# For demonstration, we'll use the mock uploader
uploader = vector_db_uploader.MockVectorDBUploader(embedder, output_dir="./mock_vector_db")

# Process and upload documents
success = vector_db_uploader.process_and_upload_documents(
    input_path="example_documents.json",
    collection_name="documents",
    uploader=uploader,
    text_field="content",
    id_field="id",
    clean_data=True,
    metadata_fields=["title", "author", "tags"]
)
```

**Using with ChromaDB:**

```python
# Initialize ChromaDB uploader
uploader = vector_db_uploader.ChromaDBUploader(embedder, persist_directory="./chroma_db")

# Process and upload documents
vector_db_uploader.process_and_upload_documents(
    input_path="example_documents.json",
    collection_name="documents",
    uploader=uploader,
    text_field="content",
    id_field="id",
    clean_data=True,
    metadata_fields=["title", "author", "tags"]
)
```

**Using with Pinecone:**

```python
# Initialize Pinecone uploader
uploader = vector_db_uploader.PineconeUploader(
    embedder=embedder,
    api_key="your_pinecone_api_key",
    environment="your_pinecone_environment"
)

# Process and upload documents
vector_db_uploader.process_and_upload_documents(
    input_path="example_documents.json",
    collection_name="documents",
    uploader=uploader,
    text_field="content",
    id_field="id",
    clean_data=True,
    metadata_fields=["title", "author", "tags"]
)
```

**Running the Example:**

```bash
python vector_db_uploader.py
```

## Complete Workflow Example

Here's an example of a complete workflow using all three scripts:

```python
import json_handler
import data_cleaner
import vector_db_uploader

# 1. Read and preprocess the JSON data
raw_data = json_handler.read_json_file("messy_documents.json")

# 2. Load into a DataFrame and clean
df = data_cleaner.load_data_to_dataframe("messy_documents.json")
df = data_cleaner.normalize_column_names(df)
df = data_cleaner.remove_duplicates(df)
df = data_cleaner.clean_text_data(df, text_columns=["content"])
df = data_cleaner.handle_missing_values(df, strategy="drop")

# 3. Save the cleaned data
data_cleaner.save_processed_data(df, "cleaned_documents.json", format="json")

# 4. Initialize embedder and uploader
embedder = vector_db_uploader.DocumentEmbedder()
uploader = vector_db_uploader.MockVectorDBUploader(embedder, output_dir="./mock_vector_db")

# 5. Upload to vector database
vector_db_uploader.process_and_upload_documents(
    input_path="cleaned_documents.json",
    collection_name="documents",
    uploader=uploader,
    text_field="content",
    id_field="id",
    clean_data=False,  # Already cleaned
    metadata_fields=["title", "author", "tags"]
)
```

## Tips for Working with Vector Databases

1. **Data Quality**: Ensure your documents are well-formatted and cleaned before generating embeddings.

2. **Embedding Models**: Choose an appropriate embedding model based on your use case. The default model (`all-MiniLM-L6-v2`) is a good starting point, but you may want to use domain-specific models for specialized applications.

3. **Vector Database Selection**: Consider your specific requirements when choosing a vector database:
   - **ChromaDB**: Good for local development and smaller datasets
   - **Pinecone**: Excellent for production deployments with large datasets
   - **Others**: Milvus, Weaviate, and Qdrant are also popular options

4. **Performance Optimization**: For large document collections, consider:
   - Batch processing documents
   - Using more efficient embedding models
   - Optimizing database configuration parameters

## Troubleshooting

- **Missing Dependencies**: If you encounter errors related to missing packages, install them using pip.
- **Memory Issues**: For large datasets, consider processing documents in batches.
- **Embedding Errors**: Ensure your text data is properly cleaned and formatted.

## Next Steps

1. Implement a search interface to query the vector database
2. Add support for additional document formats (PDF, DOCX, etc.)
3. Enhance the embedding generation with more advanced models
4. Implement document chunking for handling long documents

## License

This project is licensed under the MIT License - see the LICENSE file for details.