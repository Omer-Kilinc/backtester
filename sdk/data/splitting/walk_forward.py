import numpy as np
import pandas as pd
from typing import List, Tuple
from utils.logger import get_logger

# TODO Ensure correctness of code

from sdk.configs.data.walk_forward import WalkForwardSplitConfig

def walk_forward_split(data, config: WalkForwardSplitConfig):
    """
    Performs walk-forward splitting on time series data.
    
    Args:
        data (pd.DataFrame): Time-series data (assumed to be sorted by timestamp).
        train_window (int): Number of rows in each training window.
        test_window (int): Number of rows in each testing window after each training window.
        step_size (int): How many rows to shift the window forward each time.
    
    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (training_set, testing_set) pairs.
    """
    logger = get_logger()
    
    # Validate input parameters
    if data is None or not isinstance(data, pd.DataFrame):
        logger.error("Input data must be a pandas DataFrame.")
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if data.empty:
        logger.error("Input DataFrame is empty.")
        raise ValueError("Input DataFrame is empty.")
    
    # Initialize empty list to hold (train, test) pairs
    splits = []
    total_rows = len(data)
    start_index = 0
    
    # Check if we can create at least one split
    if config.train_window + config.test_window > total_rows:
        logger.error(f"Cannot create splits: train_window ({config.train_window}) + test_window ({config.test_window}) = {config.train_window + config.test_window} > total_rows ({total_rows})")
    
    # Perform walk-forward splitting
    while (start_index + config.train_window + config.test_window) <= total_rows:
        train_start = start_index
        train_end = start_index + config.train_window
        test_end = train_end + config.test_window
        
        # Extract training and testing sets
        training_set = data.iloc[train_start:train_end].reset_index(drop=True)
        testing_set = data.iloc[train_end:test_end].reset_index(drop=True)
        
        # Add the split to our list
        splits.append((training_set, testing_set))
        
        # Roll window forward
        start_index = start_index + config.step_size
    
    logger.info(f"Created {len(splits)} walk-forward splits with train_window={config.train_window}, test_window={config.test_window}, step_size={config.step_size}")
    
    if len(splits) == 0:
        logger.warning("No splits were created. Consider reducing window sizes or step size.")
    
    return splits