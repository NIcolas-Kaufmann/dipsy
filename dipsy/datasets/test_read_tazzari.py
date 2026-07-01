#!/usr/bin/env python3
"""
Test script to read the Tazzari2021.txt file into a pandas DataFrame
"""

import pandas as pd
import os

def read_tazzari_data():
    """Read the Tazzari2021.txt file into a pandas DataFrame"""
    
    # Get the current directory and file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'Tazzari2021.txt')
    
    print(f"Reading file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    # Read the file line by line for custom parsing
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract column names from the header line (line 8, 0-indexed)
    header_line = lines[8].strip()
    columns = [col.strip() for col in header_line.split('|') if col.strip()]
    print(f"Columns: {columns}")
    
    # Parse data lines starting from line 12 (0-indexed)
    data = []
    for i, line in enumerate(lines[12:], start=12):
        if line.strip():  # Skip empty lines
            # Use fixed-width parsing based on header alignment
            name = line[0:25].strip()
            other_name = line[25:36].strip() if len(line) > 36 else ''
            
            # Extract numeric part (everything after position 36)
            numeric_part = line[36:].strip() if len(line) > 36 else ''
            
            if numeric_part:
                numeric_values = numeric_part.split()
                
                # Handle cases where Other_Name might have spaces
                if len(numeric_values) == 11:  # Expected: 10 numeric + 1 extra from Other_Name
                    # Standard case: Name, Other_Name, 10 numeric values
                    row = [name, other_name] + numeric_values
                elif len(numeric_values) == 10 and not other_name:
                    # Case where Other_Name is empty
                    row = [name, ''] + numeric_values
                elif len(numeric_values) > 11:
                    # Case where Other_Name has spaces - take last 10 as numeric
                    extra_name_parts = numeric_values[:-10]
                    if other_name:
                        other_name = other_name + ' ' + ' '.join(extra_name_parts)
                    else:
                        other_name = ' '.join(extra_name_parts)
                    row = [name, other_name] + numeric_values[-10:]
                else:
                    print(f"Skipping malformed line {i}: {len(numeric_values)} numeric values")
                    continue
                
                if len(row) == 12:
                    data.append(row)
                else:
                    print(f"Skipping line {i}: {len(row)} fields instead of 12")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns (all except Name and Other_Name)
    numeric_cols = df.columns[2:]  # skip first two string columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == "__main__":
    try:
        # Read the data
        df = read_tazzari_data()
        
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        
        print(f"\nFirst few rows:")
        print(df.head())
        
        print(f"\nBasic statistics for numeric columns:")
        print(df.describe())
        
        # Check for any missing values
        print(f"\nMissing values per column:")
        print(df.isnull().sum())
        
        print("\n✅ Successfully read Tazzari2021.txt into pandas DataFrame!")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
