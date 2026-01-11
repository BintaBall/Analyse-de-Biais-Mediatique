import pandas as pd
from clean_text import clean_text

INPUT = "data/processed/news.csv"
OUTPUT = "data/processed/news_clean.csv"

df = pd.read_csv(INPUT)
df["clean_text"] = df["text"].apply(clean_text)

df.to_csv(OUTPUT, index=False)
print("✅ Dataset nettoyé")
