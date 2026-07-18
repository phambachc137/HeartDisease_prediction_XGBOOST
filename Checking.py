import pandas as pd

df = pd.read_csv("heart2.csv")

print(df["HeartDisease"].value_counts())

print(df["HeartDisease"].value_counts(normalize=True))


duplicates = df.duplicated().sum()
print(f"Số dòng bị trùng: {duplicates}")   