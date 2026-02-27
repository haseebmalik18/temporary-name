import pandas as pd

df = pd.read_csv("cleaned_file.csv")

print("Null values per column:")
print(df.isnull().sum())

print("\nColumns with nulls only:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nTotal null values:")
print(df.isnull().sum().sum())