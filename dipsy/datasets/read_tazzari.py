#!/usr/bin/env python3
"""
Complete working solution to read Tazzari2021.txt into a pandas DataFrame.

The file contains astronomical data in a custom format with:
- Metadata lines starting with backslashes
- Pipe-delimited headers  
- Fixed-width data with some object names containing spaces

Usage:
    df = read_tazzari_data()
"""

import pandas as pd
import os

def read_tazzari_data():
    """
    Read the Tazzari2021.txt file into a pandas DataFrame.
    
    Returns:
        pandas.DataFrame: DataFrame with 12 columns containing the astronomical data
    """
    
    # Get file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'Tazzari2021.txt')
    
    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract column names from header line (line index 8)
    header_line = lines[8].strip()
    columns = [col.strip() for col in header_line.split('|') if col.strip()]
    
    # Parse data lines starting from line index 12
    data = []
    for line in lines[12:]:
        if line.strip():  # Skip empty lines
            # Extract name (first 25 characters)
            name = line[0:25].strip()
            
            # Parse the rest of the line
            rest_of_line = line[25:].strip()
            parts = rest_of_line.split()
            
            # Find where numeric data starts (first part that can be converted to float)
            numeric_start_idx = 0
            for j, part in enumerate(parts):
                try:
                    float(part)
                    numeric_start_idx = j
                    break
                except ValueError:
                    continue
            
            # Everything before numeric data is the other_name
            other_name_parts = parts[:numeric_start_idx]
            other_name = ' '.join(other_name_parts) if other_name_parts else ''
            
            # Get the 10 numeric values
            numeric_values = parts[numeric_start_idx:numeric_start_idx + 10]
            
            if len(numeric_values) == 10:
                row = [name, other_name] + numeric_values
                data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns to proper data types
    numeric_cols = df.columns[2:]  # All columns except Name and Other_Name
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Example usage and test
if __name__ == "__main__":
    # Read the data
    df = read_tazzari_data()
    
    print(f"Successfully loaded {len(df)} astronomical objects")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    print(f"\nFirst few rows:")
    print(df[['Name', 'Other_Name', 'F3mm', 'Mdust', 'alpha']].head())
    
    print(f"\nObjects with alternative names:")
    named_objects = df[df['Other_Name'] != '']
    print(named_objects[['Name', 'Other_Name', 'F3mm', 'Mdust']])
    
    print(f"\nBasic statistics:")
    print(df[['F3mm', 'Mdust', 'alpha']].describe())
