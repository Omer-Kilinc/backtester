# This file is the main data processing pipeline for the SDK.
# The user can manually process their data, or use this standardized pipeline, which can either save data as a Parquet file send data to downsteam components.

import pandas as pd
from typing import Tuple, Union, Any

from sdk.configs.data.data_pipeline import DataPipelineConfig, APIInputConfig, SimulatedInputConfig
from sdk.data.imports.api_handler import APIImportHandler
from sdk.data.processing.transformer import DataTransformer
from sdk.data.processing.cleaning import DataCleaner
from sdk.data.price_simulation.gbm_simulator import GeometricBrownianMotionSimulator
from sdk.data.processing.noise_generator import NoiseGenerator
from sdk.data.splitting.data_splitter import split_data_by_time
from sdk.data.processing.data_conversion import DataConverter
from utils.logger import get_logger

logger = get_logger(__name__)

class DataPipeline():
    """
    The main data processing pipeline for the SDK.
    """
    def __init__(self, config: DataPipelineConfig):
        self.config = config

    def run_standard_pipeline(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], str, None]:
        """
        Runs the standard data processing pipeline.
        """
        # TODO: Implement customization of pipeline, DO NOT IGNORE
        logger.info("Starting the standard data processing pipeline...")

        data: pd.DataFrame

        # Step 1: Data Ingestion (API or Simulation)
        if isinstance(self.config.input_config, APIInputConfig):
            logger.info("Executing API import pathway.")
            # API Import
            api_handler = APIImportHandler(self.config.input_config.import_config)
            imported_data = api_handler.fetch_data()

            # Data Transformation
            transformer = DataTransformer(self.config.input_config.transformer_config)
            transformer.df = imported_data # Set the dataframe on the transformer
            transformed_data = transformer.transform()

            # Cleaning and Validation
            cleaner = DataCleaner(self.config.input_config.cleaning_config)
            cleaner.df = transformed_data # Set the dataframe on the cleaner
            data = cleaner.clean()

        elif isinstance(self.config.input_config, SimulatedInputConfig):
            logger.info("Executing simulated data pathway.")
            # Price Simulation
            simulator = GeometricBrownianMotionSimulator(self.config.input_config.price_simulator_config)
            data = simulator.simulate()
        else:
            raise TypeError("Unsupported input config type")

        # Step 2: Noise Generation (Optional)
        if self.config.noise_generator_config:
            logger.info("Applying noise generation.")
            self.config.noise_generator_config.df = data
            noise_generator = NoiseGenerator(self.config.noise_generator_config)
            data = noise_generator.inject_noise_with_gaps()

        # Step 3: Data Splitting
        logger.info("Splitting data.")
        train_df, test_df = split_data_by_time(data, self.config.data_splitter_config)
        
        # Step 4: Data Conversion (Optional)
        if self.config.data_conversion_config:
            logger.info("Saving data to Parquet.")
            self.config.data_conversion_config.df = data
            converter = DataConverter(self.config.data_conversion_config)
            file_path = converter.save_to_parquet()
            logger.info(f"Pipeline finished. Data saved to {file_path}")
            return file_path

        logger.info("Pipeline finished. Returning processed data.")
        if test_df.empty:
            return train_df
        return train_df, test_df