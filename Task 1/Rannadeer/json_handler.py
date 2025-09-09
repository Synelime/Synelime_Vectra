#!/usr/bin/env python3
"""
JSON Handler Script

This script provides utilities for working with JSON files, including reading,
writing, and basic manipulation functions.
"""

import json
import os
from typing import Dict, List, Any, Union, Optional


def read_json_file(file_path: str) -> Union[Dict, List]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content as dictionary or list
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
        
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            raise


def write_json_file(data: Union[Dict, List], file_path: str, pretty: bool = True) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Data to write (must be JSON serializable)
        file_path: Path where the JSON file will be saved
        pretty: Whether to format the JSON with indentation for readability
        
    Raises:
        TypeError: If the data is not JSON serializable
    """
    # Create directory if it doesn't exist
    dir_name = os.path.dirname(os.path.abspath(file_path))
    # Only try to create directory if there is one (not in root)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        if pretty:
            json.dump(data, file, indent=4, ensure_ascii=False)
        else:
            json.dump(data, file, ensure_ascii=False)


def merge_json_objects(obj1: Dict, obj2: Dict) -> Dict:
    """
    Merge two JSON objects (dictionaries).
    
    Args:
        obj1: First JSON object
        obj2: Second JSON object
        
    Returns:
        Merged dictionary (obj2 values override obj1 values for duplicate keys)
    """
    result = obj1.copy()
    result.update(obj2)
    return result


def extract_fields(data: Dict, fields: List[str]) -> Dict:
    """
    Extract specific fields from a JSON object.
    
    Args:
        data: Source JSON object
        fields: List of field names to extract
        
    Returns:
        Dictionary containing only the specified fields
    """
    return {field: data[field] for field in fields if field in data}


def flatten_json(nested_json: Dict, separator: str = '.') -> Dict:
    """
    Flatten a nested JSON object into a single level dictionary.
    
    Args:
        nested_json: Nested JSON object
        separator: Character to use when joining nested keys
        
    Returns:
        Flattened dictionary with compound keys
    """
    out = {}

    def flatten(x: Union[Dict, List], name: str = ''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + separator if name else a + separator)
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + separator)
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """
    Very basic JSON schema validation.
    For more complex validation, consider using a dedicated library like jsonschema.
    
    Args:
        data: JSON data to validate
        schema: Schema definition with required fields and their types
        
    Returns:
        True if valid, False otherwise
    """
    for field, field_type in schema.items():
        if field not in data:
            print(f"Missing required field: {field}")
            return False
        
        if not isinstance(data[field], field_type):
            print(f"Field {field} has incorrect type. Expected {field_type}, got {type(data[field])}")
            return False
    
    return True


def main():
    """
    Example usage of the JSON handling functions.
    """
    # Example data
    example_data = {
        "name": "Document 1",
        "metadata": {
            "author": "Ali",
            "date": "2023-06-15",
            "tags": ["vector", "database", "tutorial"]
        },
        "content": "This is an example document for the vector database."
    }
    
    # Write example data to a file
    write_json_file(example_data, "example_document.json")
    print("Created example_document.json")
    
    # Read it back
    loaded_data = read_json_file("example_document.json")
    print("\nLoaded data:")
    print(loaded_data)
    
    # Extract specific fields
    extracted = extract_fields(loaded_data, ["name", "content"])
    print("\nExtracted fields:")
    print(extracted)
    
    # Flatten the nested structure
    flattened = flatten_json(loaded_data)
    print("\nFlattened JSON:")
    print(flattened)


if __name__ == "__main__":
    main()