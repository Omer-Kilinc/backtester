import pandas as pd
import os
import pyarrow
from utils.logger import get_logger


# TODO Ensure correctness of code
# FIXME Fix error handling

from sdk.configs.data_conversion import DataConversionConfig

def save_dataframe_to_parquet(df, config: DataConversionConfig):
    logger = get_logger()

    # Make sure the folder we're going to save to actually exists
    if not os.path.isdir(config.folder_path):
        logger.error(f"The folder path '{config.folder_path}' does not exist.")
        return

    full_path = os.path.join(config.folder_path, config.filename)

    try:
        df.to_parquet(full_path, engine="pyarrow", index=False)
    except Exception as e:
        logger.error(f"Failed to save Parquet file. Error: {e}")

