# This file is the main data processing pipeline for the SDK.
# The user can manually process their data, or use this standardized pipeline, which can either save data as a Parquet file send data to downsteam components.

from sdk.configs.pipeline import PipelineConfig
from sdk.data.imports.api_handler import APIImportHandler
from sdk.data.imports.csv_handler import CSVImportHandler
from sdk.data.price_simulation.bootstrapped_returns import BootstrapReturnSimulator
from sdk.data.price_simulation.gbm_simulator import GeometricBrownianMotionSimulator
from sdk.data.price_simulation.ornstein_uhlenbeck_simulator import OrnsteinUhlenbeckSimulator
from sdk.data.transformer import DataTransformer
from sdk.data.cleaning import clean_and_validate_data
from sdk.data.noise_generator import inject_noise_with_gaps
from sdk.data.data_splitter import split_data_by_time
from sdk.data.data_conversion import save_dataframe_to_parquet
from utils.logger import get_logger

def process_data(config: PipelineConfig):
    logger = get_logger()
    logger.info("Starting data processing pipeline...")

    df = None

    if config.import_config:
        logger.info(f"Importing data using {config.import_config.import_type} handler...")
        try:
            if config.import_config.import_type == 'csv':
                importer = CSVImportHandler(config.import_config)
            else:
                importer = APIImportHandler(config.import_config)
            df = importer.import_data()
            logger.info("Data imported successfully.")
        except Exception as e:
            logger.error(f"Data import failed: {e}")
            return

        if config.transformer_config:
            logger.info("Transforming data...")
            try:
                transformer = DataTransformer(config.transformer_config)
                df = transformer.transform(df)
                logger.info("Data transformed successfully.")
            except Exception as e:
                logger.error(f"Data transformation failed: {e}")
                return

        if config.cleaning_config:
            logger.info("Cleaning and validating data...")
            try:
                df = clean_and_validate_data(df, config.cleaning_config)
                logger.info("Data cleaned and validated successfully.")
            except Exception as e:
                logger.error(f"Data cleaning and validation failed: {e}")
                return

    elif config.price_simulator_config:
        logger.info(f"Simulating price data using {config.price_simulator_config.simulator_type} simulator...")
        try:
            if config.price_simulator_config.simulator_type == 'bootstrap':
                simulator = BootstrapReturnSimulator(config.price_simulator_config)
            elif config.price_simulator_config.simulator_type == 'gbm':
                simulator = GeometricBrownianMotionSimulator(config.price_simulator_config)
            else:
                simulator = OrnsteinUhlenbeckSimulator(config.price_simulator_config)
            df = simulator.simulate()
            logger.info("Price data simulated successfully.")
        except Exception as e:
            logger.error(f"Price simulation failed: {e}")
            return
    else:
        logger.error("No data source specified. Please provide either an import configuration or a price simulation configuration.")
        return

    if df is None:
        logger.error("No data produced from import or simulation stage.")
        return

    if config.noise_generator_config:
        logger.info("Injecting noise...")
        try:
            df = inject_noise_with_gaps(df, config.noise_generator_config)
            logger.info("Noise injected successfully.")
        except Exception as e:
            logger.error(f"Noise injection failed: {e}")
            return

    if config.data_splitter_config:
        logger.info("Splitting data...")
        try:
            train_df, test_df = split_data_by_time(df, config.data_splitter_config)
            logger.info("Data split successfully.")
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            return

    if config.data_conversion_config:
        logger.info("Saving data...")
        try:
            save_dataframe_to_parquet(df, config.data_conversion_config)
            logger.info("Data saved successfully.")
        except Exception as e:
            logger.error(f"Data saving failed: {e}")
            return

    logger.info("Data processing pipeline finished successfully.")
    return train_df, test_df