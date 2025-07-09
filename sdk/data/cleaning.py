import pandas as pd
from utils.logger import get_logger
import numpy as np

# TODO Ensure correctness of code

# TODO Implement Critical Error cases, such as:
# - Missing required columns
# - Invalid data types 
# - Empty DataFrame after cleaning
# Decide how to handle these cases, and raise appropriate exceptions

from sdk.configs.cleaning import CleaningConfig

def clean_and_validate_data(data, config: CleaningConfig):
    """
    Clean and validate financial time series data.
    
    Args:
        data: Input DataFrame with time series data
        config: Configuration dictionary with cleaning options
    
    Returns:
        Cleaned and validated DataFrame
    
    Raises:
        ValueError: For critical validation errors
        TypeError: For invalid input types
    """

    logger = get_logger()



    if data is None:
        # TODO Implement Critical Error cases
        raise NotImplementedError
    
    if not isinstance(data, pd.DataFrame):
        # TODO Implement Critical Error cases
        raise NotImplementedError
    
    required_cols = config.required_columns
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise NotImplementedError


    initial_rows = len(df)
    df = df.dropna(subset=['timestamp'])
    if len(df) < initial_rows:
        logger.info(f"Removed {initial_rows - len(df)} rows with null timestamps")

    duplicate_timestamps = df['timestamp'].duplicated()
    if duplicate_timestamps.any():
        num_duplicates = duplicate_timestamps.sum()
        logger.info(f"Found {num_duplicates} duplicate timestamps")
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        logger.info(f"Removed duplicate timestamp rows, {len(df)} rows remaining")

    # TODO Timestamp strictly increasing

    # Validate numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    extreme_threshold = config.extreme_threshold
    
    for col in numerical_cols:
        if col != 'timestamp':  # Skip timestamps
            extreme_mask = (df[col].abs() > extreme_threshold) & df[col].notna()
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                if config.handle_extremes == 'remove':
                    df = df[~extreme_mask]
                    logger.info(f"Removed {extreme_count} rows with extreme values in '{col}' column")
                elif config.handle_extremes == 'flag':
                    logger.warning(f"Found {extreme_count} extreme values in '{col}' column (threshold: {extreme_threshold})")
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if col != 'timestamp' and df[col].notna().any():  # Skip timestamp and empty columns
            if df[col].nunique() <= 1:
                constant_cols.append(col)
    
    if df.empty:
        logger.error("DataFrame is empty after cleaning. Check your data source and cleaning configuration.")
                
    if constant_cols:
        logger.warning(f"Columns with constant values (possibly irrelevant): {constant_cols}")

    logger.info(f"Data cleaning completed. Final dataset shape: {df.shape}")
    return df
