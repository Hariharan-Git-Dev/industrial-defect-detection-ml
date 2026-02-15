import pandas as pd

df = pd.read_csv("train.csv")

print("Total rows:", len(df))
print("Number of NaN in EncodedPixels:", df["EncodedPixels"].isna().sum())
