import numpy as np
import pandas as pd
from typing import Dict, Any
from utils.logger import get_logger

# TODO Ensure correctness of code
# TODO Implement Critical Error cases, such as

from sdk.configs.data_splitter import DataSplitterConfig

def split_data_by_time(df, config: DataSplitterConfig):
    """
    Splits the DataFrame into training and testing sets based on the specified split type.

    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close'].
        split_type (str): Type of split - 'ratio' for percentage split or 'date' for date-based split.
        train_ratio (float): Ratio of the data to be used for training if split_type is 'ratio'.
        test_ratio (float): Ratio of the data to be used for testing if split_type is 'ratio'.
        train_start_date (str): Start date for the training set if split_type is 'date'.
        train_end_date (str): End date for the training set if split_type is 'date'.
        test_start_date (str): Start date for the testing set if split_type is 'date'.
        test_end_date (str): End date for the testing set if split_type is 'date'.

    Returns:
        tuple: (train_df, test_df) DataFrames for training and testing sets.
    """

    logger = get_logger()
    
    # Validate input DataFrame
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Input data must be a non-empty pandas DataFrame.")

    if df.empty:
        logger.error("Input DataFrame is empty.")

    required_columns = ['timestamp', 'open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")

    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')

    # Check for any failed conversions
    if df_copy['timestamp'].isnull().any():
        logger.error("Some timestamps could not be converted to datetime.")

    # Sort by timestamp
    df_copy = df_copy.sort_values(by='timestamp').reset_index(drop=True)

    if config.split_type == 'ratio':
        split_index = int(len(df_copy) * config.train_ratio)
        train_df = df_copy.iloc[:split_index].reset_index(drop=True)
        test_df = df_copy.iloc[split_index:].reset_index(drop=True)
        logger.info(f"Split data by ratio: {len(train_df)} training samples, {len(test_df)} testing samples")

    elif config.split_type == 'date':
        train_df = df_copy[(df_copy['timestamp'] >= config.train_start_date) & (df_copy['timestamp'] <= config.train_end_date)].reset_index(drop=True)
        test_df = df_copy[(df_copy['timestamp'] >= config.test_start_date) & (df_copy['timestamp'] <= config.test_end_date)].reset_index(drop=True)

        if train_df.empty:
            logger.warning("Training set is empty with the specified date range.")
        if test_df.empty:
            logger.warning("Testing set is empty with the specified date range.")

        logger.info(f"Split data by date: {len(train_df)} training samples, {len(test_df)} testing samples")

    return train_df, test_df