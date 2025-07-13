import pandas as pd
from typing import Optional, Dict
from pydantic import ValidationError

from sdk.configs.data.processing.transformer import DataTransformerConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformer:
    """Transforms DataFrame columns to a standardized format."""

    def __init__(self, config: DataTransformerConfig):
        self.config = config
        self.df = config.df.copy()

    def transform(self) -> pd.DataFrame:
        logger.info("Starting data transformation...")
        if self.config.keyword_detection:
            self._auto_detect_mapping()
        
        return self._apply_mapping()

    def _apply_mapping(self) -> pd.DataFrame:
        logger.info(f"Applying mapping: {self.config.user_mapping}")
        standardized_df = pd.DataFrame()
        
        required_fields = ["timestamp", "open", "close"]
        all_fields = required_fields + ["high", "low", "volume"]

        for std_col in all_fields:
            user_col = self.config.user_mapping.get(std_col)
            if user_col and user_col in self.df.columns:
                standardized_df[std_col] = self.df[user_col]
            elif std_col in required_fields:
                raise ValueError(f"Missing required field '{std_col}' in mapping or DataFrame.")

        # Standardize timestamp format
        if 'timestamp' in standardized_df:
            standardized_df['timestamp'] = pd.to_datetime(standardized_df['timestamp'], errors='coerce', utc=True)
            if standardized_df['timestamp'].isnull().any():
                raise ValueError("Timestamp conversion resulted in null values. Check date format.")

        # Add unmapped columns
        mapped_cols = set(self.config.user_mapping.values())
        for col in self.df.columns:
            if col not in mapped_cols:
                standardized_df[col] = self.df[col]

        logger.info("Data transformation completed.")
        return standardized_df

    def _auto_detect_mapping(self):
        logger.info("Auto-detecting column mapping...")
        column_keywords = {
            "timestamp": ["timestamp", "time", "date"],
            "open": ["open"],
            "high": ["high"],
            "low": ["low"],
            "close": ["close", "closing"],
            "volume": ["volume", "vol"]
        }
        mapping = {}
        used_columns = set()

        for std_col, keywords in column_keywords.items():
            for col in self.df.columns:
                if col.lower() in keywords and col not in used_columns:
                    mapping[std_col] = col
                    used_columns.add(col)
                    break
        
        self.config.user_mapping = mapping
        logger.info(f"Auto-detected mapping: {mapping}")


def transform_data(
    *,
    df: pd.DataFrame,
    user_mapping: Optional[Dict[str, str]] = None,
    keyword_detection: bool = True
) -> pd.DataFrame:
    """
    User-facing function to standardize DataFrame columns.

    Args:
        df: The input pandas DataFrame.
        user_mapping: An optional dictionary to map user columns to standard columns.
        keyword_detection: If True, automatically detects columns based on keywords.

    Returns:
        A new DataFrame with standardized columns.

    Raises:
        pydantic.ValidationError: If any of the parameters fail validation.
        ValueError: If mapping is invalid or required columns are missing.
    """
    try:
        config = DataTransformerConfig(
            df=df,
            user_mapping=user_mapping,
            keyword_detection=keyword_detection
        )
        transformer = DataTransformer(config)
        return transformer.transform()
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data transformation: {e}", exc_info=True)
        raise