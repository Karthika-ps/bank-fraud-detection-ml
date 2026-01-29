import os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "data")
DATA_FILE = "creditcard.csv"

def load_data():
    path = os.path.join(DATA_DIR, DATA_FILE)
    return pd.read_csv(path)
