import pandas as pd

print("Starting test...")

# Simple test to read the file
df = pd.read_csv('Tazzari2021.txt', 
                 sep='|', 
                 skiprows=9, 
                 skipinitialspace=True)

print(f"Shape: {df.shape}")
print("First few lines:")
print(df.head())
print("Done!")
