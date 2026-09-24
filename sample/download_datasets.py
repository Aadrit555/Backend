import pandas as pd
from sklearn.datasets import fetch_california_housing
from datasets import load_dataset
import json
import os

os.makedirs('sample', exist_ok=True)

print("Downloading ML dataset (California Housing)...")
try:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv('sample/california_housing.csv', index=False)
    print(f"Saved sample/california_housing.csv with shape {df.shape}")
except Exception as e:
    print(f"Failed to download ML dataset: {e}")

print("Downloading RAG dataset (Wikitext-2)...")
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    count = 0
    with open('sample/wikitext_rag.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            text = item['text'].strip()
            if text:  # Skip empty lines
                f.write(json.dumps({"text": text}) + "\n")
                count += 1
                if count >= 20000:  # Save 20k rows for a good sized dataset
                    break
    print(f"Saved {count} lines to sample/wikitext_rag.jsonl")
except Exception as e:
    print(f"Failed to download RAG dataset: {e}")
