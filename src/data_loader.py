import pandas as pd
import numpy as np
from typing import Tuple

def load_and_split_data(filepath: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw dataset and performs a stratified-like random split into train and test sets.
    
    Args:
        filepath (str): Path to the raw CSV data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generator to ensure reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training dataframe and Testing dataframe.
    """
    df = pd.read_csv(filepath)
    
    # Ensure reproducibility
    np.random.seed(random_state)
    
    # Generate random permutation of indices
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, test_df