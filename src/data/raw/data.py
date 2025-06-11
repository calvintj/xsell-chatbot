import os

import pandas as pd
from datasets import load_dataset


def get_dataset():
    csv_path = "llama2_papers.csv"

    # Check if CSV file exists
    if os.path.exists(csv_path):
        print("Loading dataset from CSV...")
        return pd.read_csv(csv_path)

    # If CSV doesn't exist, download and save it
    print("Downloading dataset...")
    dataset = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")
    df = dataset.to_pandas()

    # Save to CSV
    print("Saving dataset to CSV...")
    df.to_csv(csv_path, index=False)

    return df


# Load the dataset
panda = get_dataset()
