from utils.logger import get_logger  
import pandas as pd
from sdk.configs.imports.csv_handler import CSVImportConfig

# TODO Ensure correctness of code

class CSVImportHandler:
    def __init__(self, config: CSVImportConfig):
        self.config = config

    def load_raw_dataframe(self):
        logger = get_logger()  
        try:
            df = pd.read_csv(self.config.filepath)
            logger.info(f"Loaded CSV with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return None
