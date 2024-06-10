import requests
from io import BytesIO
from typing import List
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    urls_to_try = [
        'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    ]

    for url in urls_to_try:
        response = requests.get(url)
        
        if response.status_code == 200:
            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)
            print(f"Successfully downloaded: {url}")
        else:
            print(f"Failed to download: {url}, Status Code: {response.status_code}")

    if not dfs:
        raise Exception("No data files were successfully downloaded.")
    
    return pd.concat(dfs, ignore_index=True)
