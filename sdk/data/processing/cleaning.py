import pandas as pd
import numpy as np
from typing import List, Literal
from pydantic import ValidationError

from utils.logger import get_logger
from sdk.configs.data.processing.cleaning import CleaningConfig

logger = get_logger(__name__)


class DataCleaner:
    """Cleans and validates financial time series data based on a configuration."""

    def __init__(self, config: CleaningConfig):
        self.config = config
        self.df = config.data.copy()

    def clean(self) -> pd.DataFrame:
        logger.info("Starting data cleaning process...")
        initial_rows = len(self.df)

        self._handle_timestamps()
        self._fill_nulls()
        self._handle_extremes()
        self._check_for_constant_columns()

        if self.df.empty:
            logger.error("DataFrame is empty after cleaning.")
            raise ValueError("DataFrame became empty after cleaning process.")

        final_rows = len(self.df)
        logger.info(f"Data cleaning completed. Rows changed from {initial_rows} to {final_rows}.")
        return self.df

    def _handle_timestamps(self):
        df = self.df
        # Ensure timestamp is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Remove rows with null timestamps
        initial_rows = len(df)
        df.dropna(subset=['timestamp'], inplace=True)
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows with null timestamps.")

        # Remove duplicate timestamps
        if df['timestamp'].duplicated().any():
            num_duplicates = df['timestamp'].duplicated().sum()
            df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            logger.info(f"Removed {num_duplicates} duplicate timestamps.")

        # Sort by timestamp and check for strict increase
        df.sort_values('timestamp', inplace=True)
        if self.config.strict_timestamps and not df['timestamp'].is_monotonic_increasing:
            logger.error("Timestamps are not strictly increasing after sorting.")
            raise ValueError("Timestamps must be strictly increasing.")
        self.df = df

    def _fill_nulls(self):
        if self.config.fill_nulls:
            self.df.fillna(method='ffill', inplace=True)
            self.df.fillna(method='bfill', inplace=True) # Fill any remaining NaNs at the beginning
            logger.info("Filled null values using forward-fill and backward-fill.")

    def _handle_extremes(self):
        if self.config.handle_extremes == 'ignore':
            return

        numerical_cols = self.df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            if col != 'timestamp':
                extreme_mask = self.df[col].abs() > self.config.extreme_threshold
                extreme_count = extreme_mask.sum()

                if extreme_count > 0:
                    if self.config.handle_extremes == 'remove':
                        self.df = self.df[~extreme_mask]
                        logger.info(f"Removed {extreme_count} rows with extreme values in '{col}'.")
                    elif self.config.handle_extremes == 'flag':
                        logger.warning(f"Found {extreme_count} extreme values in '{col}'.")

    def _check_for_constant_columns(self):
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_cols:
            logger.warning(f"Columns with constant values detected: {constant_cols}")


def clean_data(
    *,
    data: pd.DataFrame,
    fill_nulls: bool = True,
    extreme_threshold: float = 1e6,
    handle_extremes: Literal['flag', 'remove', 'ignore'] = 'remove',
    strict_timestamps: bool = True,
    required_columns: List[str] = ['timestamp', 'open', 'close'],
    optional_columns: List[str] = ['high', 'low', 'volume']
) -> pd.DataFrame:
    """
    User-facing function to clean and validate financial time series data.

    This function validates inputs, initializes the cleaner, runs the cleaning process,
    and returns the cleaned pandas DataFrame.

    Args:
        data: The input pandas DataFrame with time series data.
        fill_nulls: If True, fills null values using forward and backward fill.
        extreme_threshold: The threshold for detecting extreme numerical values.
        handle_extremes: How to handle extreme values ('flag', 'remove', or 'ignore').
        strict_timestamps: If True, ensures timestamps are strictly increasing.
        required_columns: A list of columns that must be present in the DataFrame.
        optional_columns: A list of optional columns.

    Returns:
        A cleaned and validated pandas DataFrame.

    Raises:
        pydantic.ValidationError: If any of the parameters fail validation.
        ValueError: If the DataFrame is or becomes empty, or if timestamps are not unique/increasing.
    """
    try:
        config = CleaningConfig(
            data=data,
            fill_nulls=fill_nulls,
            extreme_threshold=extreme_threshold,
            handle_extremes=handle_extremes,
            strict_timestamps=strict_timestamps,
            required_columns=required_columns,
            optional_columns=optional_columns
        )
        cleaner = DataCleaner(config)
        return cleaner.clean()
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data cleaning: {e}", exc_info=True)
        raise

