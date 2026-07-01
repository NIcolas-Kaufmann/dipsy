# Easy way to read Tazzari2021.txt into pandas DataFrame

import pandas as pd

def read_tazzari_data():
    """Read Tazzari2021.txt into a pandas DataFrame"""
    
    with open('Tazzari2021.txt', 'r') as f:
        lines = f.readlines()
    
    # Get column names from header (line 8)
    header_line = lines[8].strip()
    columns = [col.strip() for col in header_line.split('|') if col.strip()]
    
    # Parse data starting from line 12
    data = []
    for line in lines[12:]:
        if line.strip():
            # Extract name (first 25 chars)
            name = line[0:25].strip()
            rest = line[25:].strip().split()
            
            # Find where numbers start
            numeric_start = 0
            for i, part in enumerate(rest):
                try:
                    float(part)
                    numeric_start = i
                    break
                except:
                    continue
            
            # Split into other_name and numeric values
            other_name = ' '.join(rest[:numeric_start]) if numeric_start > 0 else ''
            numeric_vals = rest[numeric_start:numeric_start + 10]
            
            if len(numeric_vals) == 10:
                data.append([name, other_name] + numeric_vals)
    
    # Create DataFrame and convert numeric columns
    df = pd.DataFrame(data, columns=columns)
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col])
    
    return df

# Usage example:
if __name__ == "__main__":
    df = read_tazzari_data()
    print(f"Loaded {len(df)} objects")
    print(df.head())
