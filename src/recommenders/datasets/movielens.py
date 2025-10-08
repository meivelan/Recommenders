import requests
import zipfile
import os
from tqdm import tqdm

from typing import Dict

import pandas as pd

class MovieLens:
    """
    Initialize MovieLens dataset
    """
    urls = {
        "ml-latest-small" : "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", 
        "ml-1M"   : "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-10M"  : "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "ml-20M"  : "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "ml-25M"  : "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    }
    def __init__(self, dataset_name):
        """
        Intialize the MovieLens dataset with specified version.

        Parameters:
        dataset_name : name of the dataset ("ml-latest-small" or "ml-1M" or ml-10M" or "ml-20M" or "ml-25M")
        """
        self.dataset_name = dataset_name
        self.url = MovieLens.urls[self.dataset_name]
        self.download_path = f"{self.dataset_name}.zip"
        self.extract_path = self.dataset_name
        self.links = None
        self.movies = None
        self.ratings = None
        self.tags = None

        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            
            print(f"Downloading {self.dataset_name} dataset...")
            total_size = int(response.headers.get('content-length', 0))

            with open(self.download_path, "wb") as file, tqdm(
                desc=self.download_path,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))
            print(f"Downloaded to {self.download_path}")

        except requests.exceptions.RequestException as error:
            print(f"Error: {error}, while downloading {self.dataset_name} : {error}")
            return 
    
        try:
            print(f"Extracting {self.dataset_name}...")
            with zipfile.ZipFile(self.download_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)

            print(f"Extracted files for {dataset_name}:")
            for file in os.listdir(self.extract_path):
                print(f"  - {file}")
        
        except zipfile.BadZipFile as error:
            print(f"Error: {error}, {self.download_path} is not a valid zip file")
            return 

        finally:
            if os.path.exists(self.download_path):
                os.remove(self.download_path)
                print(f"Removed temporary file {self.download_path}")
        

    def load_data(self, file)-> pd.DataFrame:
        """
        Load specified "file".

        Parameters:
        file : name of the file

        Returns:
        pd.DataFrame
        """
        try:
            return pd.read_csv(f"{self.extract_path}/{file}")
            
        except FileNotFoundError as error:
            print(f"Error: {error}")

    
    def load_all_data(self)-> Dict[str, pd.DataFrame]:
        """
        Load all the "file(s)" in the "extract_path" directory.


        Returns:
        Dict[str, pd.Dataframe]
        """
        dataset = {file :self.load_data(file) for file in os.listdir(self.extract_path)}
        return dataset

    
if __name__=="__main__":
    movielens = MovieLens("ml-latest-small")