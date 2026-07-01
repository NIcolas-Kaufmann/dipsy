#!/usr/bin/env python3
"""
Improved script to read the Tazzari2021.txt file into a pandas DataFrame
This version handles all edge cases including Other_Name fields with spaces.
"""

import pandas as pd
import os

def read_tazzari_data_improved():
    """Read the Tazzari2021.txt file into a pandas DataFrame with improved parsing"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'Tazzari2021.txt')
    
    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract column names from header line
    header_line = lines[8].strip()
    columns = [col.strip() for col in header_line.split('|') if col.strip()]
    
    data = []
    for i, line in enumerate(lines[12:], start=12):
        if line.strip():
            # Parse using fixed positions based on the original format
            # Looking at the header alignment: Name (25 chars), Other_Name (11 chars), then numbers
            name = line[0:25].strip()
            
            # Find where the numeric data starts by looking for the first float-like pattern
            rest_of_line = line[25:].strip()
            parts = rest_of_line.split()
            
            # Find the first part that looks like a float (starts with digit or decimal)
            numeric_start_idx = 0
            for j, part in enumerate(parts):
                try:
                    float(part)
                    numeric_start_idx = j
                    break
                except ValueError:
                    continue
            
            # Everything before the numeric part is Other_Name
            other_name_parts = parts[:numeric_start_idx]
            other_name = ' '.join(other_name_parts) if other_name_parts else ''
            
            # Everything from numeric_start_idx onwards are the numeric values
            numeric_values = parts[numeric_start_idx:]
            
            if len(numeric_values) >= 10:  # We expect 10 numeric columns
                # Take exactly 10 numeric values
                row = [name, other_name] + numeric_values[:10]
                data.append(row)
            else:
                print(f"Skipping line {i}: only {len(numeric_values)} numeric values found")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns
    numeric_cols = df.columns[2:]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == "__main__":
    try:
        df = read_tazzari_data_improved()
        
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nSample of data with Other_Name:")
        print(df[df['Other_Name'] != ''][['Name', 'Other_Name', 'F3mm', 'Mdust']].head())
        
        print(f"\nAll data:")
        print(df[['Name', 'Other_Name', 'F3mm', 'Mdust']].head(10))
        
        print(f"\nData types:\n{df.dtypes}")
        
        print(f"\n✅ Successfully read all {len(df)} rows from Tazzari2021.txt!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
