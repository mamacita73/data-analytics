from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"

df = pd.read_csv(RAW / "statista_2024.csv",
                 sep=';',            # WICHTIG!
                 encoding='utf-8-sig')

# optional aufräumen/vereinheitlichen
df.columns = [c.strip().lower() for c in df.columns]
df = df.rename(columns={'category': 'category', 'value': 'value'})
df['value'] = pd.to_numeric(df['value'], errors='coerce')

print(df)
