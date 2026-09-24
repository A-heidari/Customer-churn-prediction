import pandas as pd 
from pathlib import Path

def load_data(filename): 

    path_file = Path("data/raw") / filename

    df = pd.read_csv(path_file)
    return df

