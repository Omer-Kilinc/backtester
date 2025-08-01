from sdk.backtester.portfoliostate import PortfolioState
from sdk.configs.backtester.backtester import BacktesterConfig
from sdk.strategy.base import Strategy
from sdk.data.data_pipeline import DataPipelineConfig, DataPipeline
from typing import TypeVar
from logging import getLogger
from sdk.strategy.registry import INDICATOR_REGISTRY
import pandas as pd
import numpy as np

logger = getLogger(__name__)

T = TypeVar('T', bound=Strategy)

class Backtester:
    """
    Backtester class for running backtests.

    Attributes:
        strategy (Inherited from the Strategy class): The strategy to run the backtest on.
        data_pipeline_config (DataPipelineConfig): The data pipeline configuration.
        config (BacktesterConfig): The backtester configuration.
        data (pd.DataFrame): The data to run the backtest on.
    """
    def __init__(
        self, 
        strategy: T, 
        data_pipeline_config: DataPipelineConfig, 
        config: BacktesterConfig, 
        data: pd.DataFrame = None
    ):
        """
        Initialize the backtester.

        Args:
            strategy (Inherited from Strategy class): The strategy to run the backtest on.
            data_pipeline_config (DataPipelineConfig): The data pipeline configuration.
            config (BacktesterConfig): The backtester configuration.
            data (pd.DataFrame, optional): The data to run the backtest on. If not provided, the data will be prepared using the data pipeline.
        """
        # TODO: Decide whether to keep other processes like Data processing within the backtester or to keep the backtester seperate and combine them in a seperate function.

        # TODO: Pydantic or keep it like this?
        if not (isinstance(strategy, Strategy) and type(strategy) != Strategy):
            # TODO: Remember the logger error issue
            logger.error(
                f"'strategy' must be an instance of a subclass of 'Strategy', "
                f"not '{type(strategy).__name__}' or 'Strategy' directly."
            )
            
        self.strategy = strategy
        self.data_pipeline_config = data_pipeline_config
        self.config = config
        self.data = data
        self.portfoliostate=PortfolioState(
            cash=self.config.initial_capital,
            initial_capital=self.config.initial_capital,
            margin_requirement_rate=self.config.margin_requirement_rate
        )
        

    def prepare_data(self):
        """
        Prepare the data for the backtest. 
        """
        result = DataPipeline(self.data_pipeline_config).run_standard_pipeline()
        
        # Handle different return types from pipeline
        if isinstance(result, tuple):
            # If we get (train_df, test_df), use training data
            self.data = result[0]
            logger.info("Using training data from pipeline split")
        elif isinstance(result, pd.DataFrame):
            # Single DataFrame
            self.data = result
            logger.info("Using single DataFrame from pipeline")
        elif isinstance(result, str):
            # File path - need to load it
            self.data = pd.read_parquet(result)
            logger.info(f"Loaded data from file: {result}")
        else:
            raise ValueError(f"Unexpected data type from pipeline: {type(result)}")

    def execute_backtest(self):
        """
        Execute the backtest.
        """
        logger.info("Executing backtest...")
        self.strategy.on_init()
        
        if self.data is None:
            self.prepare_data()

        self.precompute_indicators(progress_log_freq=self.config.progress_log_freq)

        
    
        for i in range(len(self.data)):
            # Need to check everything that occured between the previous bar close and current bar close
            # Firstly, check if any margin requirements have been met
            current_bar_data = self.data.iloc[:i+1]

            self._process_intrabar_events(current_bar_data)
            

                    
        
        logger.info("Backtest completed.")
        self.strategy.on_teardown()
    
    def _process_intrabar_events(self, current_bar_data):

        logger.debug(f"Processing intrabar events for bar with OHLC: {current_bar_data.iloc[-1]['open']:.2f}, {current_bar_data.iloc[-1]['high']:.2f}, {current_bar_data.iloc[-1]['low']:.2f}, {current_bar_data.iloc[-1]['close']:.2f}")

        # Conserative approach involves:
        # 1. Checking for margin requirements
        
        
        # 2. Process Stop Losses
        # 3. Process Take Profits
        # 4. Process User set Exit Conditions (custom)
        # 5. Process User set entry conditions like limit orders

        return


    def precompute_indicators(self, progress_log_freq: float = 10.0):
        """
        Precompute indicators with adjustable progress logging.
        
        Args:
            progress_log_freq: Percentage (0-100) for progress updates. 
                          0 = off, 10 = every 10% (default), 100 = every indicator.
        """
        logger.info("Precomputing indicators...")
    
        indicators = {
            name: info for name, info in INDICATOR_REGISTRY.items() 
            if info.get('precompute')
        }
        
        # Initialize all columns at once
        self.data = self.data.assign(**{name: np.nan for name in indicators})
        
        # Calculate logging interval
        log_every = 1
        if 0 < progress_log_freq <= 100:
            log_every = max(1, int(len(indicators) * (progress_log_freq / 100)))
        
        for i, (name, info) in enumerate(indicators.items(), 1):
            # Progress logging (skip if freq=0)
            if progress_log_freq > 0 and (i % log_every == 0 or i == len(indicators)):
                logger.info(
                    f"Progress: {i}/{len(indicators)} "
                    f"({(i/len(indicators)*100):.1f}%) - Computing {name}"
                )
            
            # Compute indicator
            try:
                if info.get('vectorized', False):
                    self.data[name] = info['func'](self.data)
                else:
                    self.data[name] = [
                        info['func'](self.data.iloc[:j+1]) 
                        for j in range(len(self.data))
                    ]
            except Exception as e:
                logger.error(f"Failed computing {name}: {str(e)}")
                continue
                
