# This file is the main data processing pipeline for the SDK.
# The user can manually process their data, or use this standardized pipeline, which can either save data as a Parquet file send data to downsteam components.

import pandas as pd
from typing import Tuple, Union, Any, List

from sdk.configs.data.data_pipeline import DataPipelineConfig, APIInputConfig, SimulatedInputConfig
from sdk.configs.data.price_simulation.gbm_simulator import GBMSimulatorConfig
from sdk.configs.data.price_simulation.ornstein_uhlenbeck_simulator import OrnsteinUhlenbeckSimulatorConfig
from sdk.configs.data.price_simulation.bootstrapped_returns import BootstrapReturnSimulatorConfig
from sdk.data.imports.api_handler import APIImportHandler
from sdk.data.processing.transformer import DataTransformer
from sdk.data.processing.cleaning import DataCleaner
from sdk.data.price_simulation.gbm_simulator import GeometricBrownianMotionSimulator
from sdk.data.price_simulation.ornstein_uhlenbeck_simulator import OrnsteinUhlenbeckSimulator
from sdk.data.price_simulation.bootstrapped_returns import BootstrapReturnSimulator
from sdk.data.processing.noise_generator import NoiseGenerator
from sdk.data.splitting.data_splitter import split_data_by_time
from sdk.data.splitting.walk_forward import walk_forward_split
from sdk.data.processing.data_conversion import DataConverter
from sdk.configs.data.processing.data_conversion import DataConversionConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class DataPipeline():
    """
    The main data processing pipeline for the SDK.
    """
    def __init__(self, config: DataPipelineConfig):
        self.config = config

    def run_standard_pipeline(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], List[Tuple[pd.DataFrame, pd.DataFrame]], List[Tuple[str, str]], str, None]:
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
            # Price Simulation - Support multiple simulators
            sim_config = self.config.input_config.price_simulator_config
            
            if isinstance(sim_config, GBMSimulatorConfig):
                logger.info("Using Geometric Brownian Motion simulator.")
                simulator = GeometricBrownianMotionSimulator(sim_config)
            elif isinstance(sim_config, OrnsteinUhlenbeckSimulatorConfig):
                logger.info("Using Ornstein-Uhlenbeck simulator.")
                simulator = OrnsteinUhlenbeckSimulator(sim_config)
            elif isinstance(sim_config, BootstrapReturnSimulatorConfig):
                logger.info("Using Bootstrap Return simulator.")
                simulator = BootstrapReturnSimulator(sim_config)
            else:
                raise TypeError(f"Unsupported simulator config type: {type(sim_config)}")
            
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
        if self.config.walk_forward_config:
            logger.info("Performing walk-forward splitting.")
            splits = walk_forward_split(data, self.config.walk_forward_config)
            # For walk-forward, return the splits directly
            if self.config.data_conversion_config:
                logger.info("Saving walk-forward splits to Parquet files.")
                file_paths = []
                for i, (train_split, test_split) in enumerate(splits):
                    # Save each split pair with modified filenames
                    base_filename = self.config.data_conversion_config.filename.replace('.parquet', '')
                    
                    # Create train split config
                    train_config = DataConversionConfig(
                        df=train_split,
                        folder_path=self.config.data_conversion_config.folder_path,
                        filename=f"{base_filename}_wf_train_{i}.parquet"
                    )
                    train_converter = DataConverter(train_config)
                    train_path = train_converter.save_to_parquet()
                    
                    # Create test split config
                    test_config = DataConversionConfig(
                        df=test_split,
                        folder_path=self.config.data_conversion_config.folder_path,
                        filename=f"{base_filename}_wf_test_{i}.parquet"
                    )
                    test_converter = DataConverter(test_config)
                    test_path = test_converter.save_to_parquet()
                    
                    file_paths.append((train_path, test_path))
                
                logger.info(f"Walk-forward pipeline finished. {len(file_paths)} split pairs saved.")
                return file_paths
            else:
                logger.info("Walk-forward pipeline finished. Returning split data.")
                return splits
        else:
            logger.info("Splitting data using standard method.")
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