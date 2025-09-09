# Running Instructions for Vector Database Document Upload Scripts

## Prerequisites

Before running the scripts, make sure you have installed all the required dependencies:

```bash
pip install pandas numpy sentence-transformers chromadb pinecone-client
```

## Running Individual Scripts

### 1. JSON Handler

The `json_handler.py` script provides utilities for working with JSON files. To run it:

```bash
python json_handler.py
```

This will demonstrate basic JSON operations like reading, writing, and manipulating JSON data.

### 2. Data Cleaner

The `data_cleaner.py` script provides utilities for cleaning and preprocessing document data. To run it:

```bash
python data_cleaner.py
```

This will demonstrate data cleaning operations like removing duplicates, handling missing values, and extracting text features.

### 3. Vector DB Uploader

The `vector_db_uploader.py` script provides utilities for uploading documents to vector databases. To run it:

```bash
python vector_db_uploader.py
```

This will demonstrate document embedding generation and uploading to a mock vector database.

## Running the Complete Workflow

The `complete_workflow.py` script demonstrates how to use all three components together in a single workflow. To run it:

```bash
python complete_workflow.py
```

This will:
1. Load and process JSON data using `json_handler.py`
2. Clean and preprocess the data using `data_cleaner.py`
3. Generate embeddings and upload to a mock vector database using `vector_db_uploader.py`

## Troubleshooting

### Pinecone Import Error

If you encounter an error with Pinecone import, make sure you have the latest version of the Pinecone client:

```bash
pip install --upgrade pinecone-client
```

The scripts have been updated to work with the latest Pinecone client API.

### Missing Dependencies

If you encounter import errors, make sure you have installed all the required dependencies as mentioned in the Prerequisites section.

### File Not Found Errors

Make sure you are running the scripts from the correct directory where all the script files are located.