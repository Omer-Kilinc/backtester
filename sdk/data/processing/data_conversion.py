import pandas as pd
import os
from pydantic import ValidationError

from utils.logger import get_logger
from sdk.configs.data.processing.data_conversion import DataConversionConfig

logger = get_logger(__name__)


class DataConverter:
    """Handles the conversion and saving of DataFrames to different formats."""

    def __init__(self, config: DataConversionConfig):
        self.config = config

    def save_to_parquet(self) -> str:
        """
        Saves the DataFrame to a Parquet file.

        Returns:
            The full path to the saved file.
        """
        full_path = os.path.join(self.config.folder_path, self.config.filename)
        logger.info(f"Attempting to save DataFrame to Parquet file at: {full_path}")

        try:
            self.config.df.to_parquet(full_path, engine="pyarrow", index=False)
            logger.info(f"Successfully saved Parquet file to: {full_path}")
            return full_path
        except Exception as e:
            logger.error(f"Failed to save Parquet file to '{full_path}'. Error: {e}", exc_info=True)
            raise IOError(f"Failed to save Parquet file due to: {e}") from e


def save_to_parquet(
    *,
    df: pd.DataFrame,
    folder_path: str,
    filename: str = "simulated_data.parquet"
) -> str:
    """
    User-facing function to save a pandas DataFrame to a Parquet file.

    This function validates inputs, initializes the converter, saves the file,
    and returns the full path to the saved file.

    Args:
        df: The pandas DataFrame to save.
        folder_path: The directory where the file will be saved.
        filename: The name of the output file ('.parquet' extension is added if missing).

    Returns:
        The full path to the saved Parquet file.

    Raises:
        pydantic.ValidationError: If any of the parameters fail validation.
        IOError: If the file cannot be saved.
    """
    try:
        config = DataConversionConfig(
            df=df,
            folder_path=folder_path,
            filename=filename
        )
        converter = DataConverter(config)
        return converter.save_to_parquet()
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during file conversion: {e}", exc_info=True)
        raise

