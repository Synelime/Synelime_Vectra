#!/usr/bin/env python3
"""
Data Cleaning Script

This script provides utilities for cleaning and preprocessing document data
using Pandas and NumPy before uploading to a vector database.
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any, Union, Optional
from pathlib import Path


def load_data_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load data from a JSON file into a pandas DataFrame.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        DataFrame containing the data
    """
    # Determine file type from extension
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.json':
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # If it's a dictionary of records
            if any(isinstance(v, dict) for v in data.values()):
                return pd.DataFrame.from_dict(data, orient='index')
            # If it's a single record
            else:
                return pd.DataFrame([data])
    elif file_ext in ['.csv', '.txt']:
        return pd.read_csv(file_path)
    elif file_ext == '.xlsx':
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Column names to consider for identifying duplicates
              (if None, all columns are used)
              
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_values: Dict = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ('drop', 'fill')
        fill_values: Dictionary mapping column names to fill values
                    (used only if strategy is 'fill')
                    
    Returns:
        DataFrame with missing values handled
    """
    if strategy == 'drop':
        return df.dropna().reset_index(drop=True)
    
    elif strategy == 'fill':
        if fill_values is None:
            fill_values = {}
            
        df_copy = df.copy()
        
        # Fill specified columns with provided values
        for col, value in fill_values.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(value)
        
        # For remaining columns with NaN, use appropriate defaults
        for col in df_copy.columns:
            if col not in fill_values and df_copy[col].isna().any():
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].fillna(0)
                elif pd.api.types.is_string_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].fillna('')
                else:
                    # For other types, fill with the most common value
                    most_common = df_copy[col].mode()[0] if not df_copy[col].mode().empty else None
                    df_copy[col] = df_copy[col].fillna(most_common)
        
        return df_copy
    
    else:
        raise ValueError(f"Unsupported strategy: {strategy}. Use 'drop' or 'fill'.")


def clean_text_data(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Clean text data in specified columns.
    
    Args:
        df: Input DataFrame
        text_columns: List of column names containing text data
        
    Returns:
        DataFrame with cleaned text data
    """
    df_copy = df.copy()
    
    for col in text_columns:
        if col in df_copy.columns:
            # Convert to string type if not already
            df_copy[col] = df_copy[col].astype(str)
            
            # Remove extra whitespace
            df_copy[col] = df_copy[col].str.strip()
            df_copy[col] = df_copy[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
            df_copy[col] = df_copy[col].str.replace(r'[^\w\s.,;:!?-]', '', regex=True)
            
            # Replace empty strings with NaN
            df_copy[col] = df_copy[col].replace('', np.nan)
    
    return df_copy


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to a consistent format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df_copy = df.copy()
    
    # Convert to lowercase, replace spaces with underscores
    df_copy.columns = [col.lower().replace(' ', '_') for col in df_copy.columns]
    
    # Remove special characters
    df_copy.columns = [re.sub(r'[^\w_]', '', col) for col in df_copy.columns]
    
    return df_copy


def convert_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Convert data types of columns according to the provided mapping.
    
    Args:
        df: Input DataFrame
        type_mapping: Dictionary mapping column names to desired data types
                     (e.g., {'age': 'int', 'price': 'float', 'date': 'datetime'})
                     
    Returns:
        DataFrame with converted data types
    """
    df_copy = df.copy()
    
    for col, dtype in type_mapping.items():
        if col in df_copy.columns:
            try:
                if dtype == 'int':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)
                elif dtype == 'float':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif dtype == 'datetime':
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                elif dtype == 'string' or dtype == 'str':
                    df_copy[col] = df_copy[col].astype(str)
                elif dtype == 'category':
                    df_copy[col] = df_copy[col].astype('category')
                elif dtype == 'bool':
                    df_copy[col] = df_copy[col].astype(bool)
            except Exception as e:
                print(f"Error converting column {col} to {dtype}: {e}")
    
    return df_copy


def extract_features(df: pd.DataFrame, text_column: str, max_features: int = 100) -> pd.DataFrame:
    """
    Extract basic text features from a text column.
    
    Args:
        df: Input DataFrame
        text_column: Name of the column containing text data
        max_features: Maximum number of features to extract
        
    Returns:
        DataFrame with additional feature columns
    """
    if text_column not in df.columns:
        raise ValueError(f"Column {text_column} not found in DataFrame")
    
    df_copy = df.copy()
    
    # Add character count
    df_copy[f"{text_column}_char_count"] = df_copy[text_column].str.len()
    
    # Add word count
    df_copy[f"{text_column}_word_count"] = df_copy[text_column].str.split().str.len()
    
    # Add sentence count
    df_copy[f"{text_column}_sentence_count"] = df_copy[text_column].str.count(r'[.!?]+')
    
    # Add average word length
    df_copy[f"{text_column}_avg_word_length"] = df_copy[text_column].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    return df_copy


def save_processed_data(df: pd.DataFrame, output_path: str, format: str = 'json') -> None:
    """
    Save the processed DataFrame to a file.
    
    Args:
        df: DataFrame to save
        output_path: Path where the file will be saved
        format: Output file format ('json', 'csv', 'excel')
    """
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        df.to_json(output_path, orient='records', indent=4)
    elif format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format.lower() == 'excel':
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}. Use 'json', 'csv', or 'excel'.")


def main():
    """
    Example usage of the data cleaning functions.
    """
    # Example: Create a sample DataFrame
    data = [
        {"title": "Document 1", "content": "This is   an example document.", "tags": "vector,database", "date": "2023-06-15"},
        {"title": "Document 2", "content": "Another example with missing data", "tags": np.nan, "date": "2023-06-16"},
        {"title": "Document 1", "content": "This is   an example document.", "tags": "vector,database", "date": "2023-06-15"},  # Duplicate
        {"title": "Document 3", "content": np.nan, "tags": "tutorial,guide", "date": "invalid_date"},
    ]
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    # 1. Normalize column names
    df = normalize_column_names(df)
    
    # 2. Remove duplicates
    df = remove_duplicates(df)
    
    # 3. Clean text data
    df = clean_text_data(df, text_columns=['content'])
    
    # 4. Handle missing values
    df = handle_missing_values(df, strategy='fill', fill_values={'content': 'No content available'})
    
    # 5. Convert data types
    df = convert_data_types(df, type_mapping={'date': 'datetime'})
    
    # 6. Extract features from text
    df = extract_features(df, text_column='content')
    
    print("Cleaned DataFrame:")
    print(df)
    
    # Save the processed data
    save_processed_data(df, "processed_documents.json")
    print("\nSaved processed data to processed_documents.json")


if __name__ == "__main__":
    main()