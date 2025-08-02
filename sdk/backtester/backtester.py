from sdk.backtester.portfoliostate import PortfolioState
from sdk.configs.backtester.backtester import BacktesterConfig
from sdk.strategy.base import Strategy
from sdk.data.data_pipeline import DataPipelineConfig, DataPipeline
from sdk.configs.backtester.tradeinstruction import OrderDirection
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
        if not (isinstance(strategy, Strategy) and type(strategy) != Strategy):
            logger.error(
                f"'strategy' must be an instance of a subclass of 'Strategy', "
                f"not '{type(strategy).__name__}' or 'Strategy' directly."
            )
            
        self.strategy = strategy
        self.data_pipeline_config = data_pipeline_config
        self.config = config
        self.data = data
        self.portfoliostate = PortfolioState(
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
            current_bar_data = self.data.iloc[:i+1]
            self._process_intrabar_events(current_bar_data)
        
        logger.info("Backtest completed.")
        self.strategy.on_teardown()

    def _process_intrabar_events(self, current_bar_data):
        """Process all intrabar events in priority order"""
        logger.debug(f"Processing intrabar events for bar with OHLC: {current_bar_data.iloc[-1]['open']:.2f}, {current_bar_data.iloc[-1]['high']:.2f}, {current_bar_data.iloc[-1]['low']:.2f}, {current_bar_data.iloc[-1]['close']:.2f}")

        # Conservative approach:
        # 1. Checking for margin requirements
        self._check_margin_requirements(current_bar_data)
        
        # 2. Process Stop Losses
        # 3. Process Take Profits
        # 4. Process User set Exit Conditions (custom)
        # 5. Process User set entry conditions like limit orders

    def _check_margin_requirements(self, current_bar_data: pd.DataFrame):
        """
        Check if any margin requirements are violated and liquidate if necessary.
        Uses sophisticated equity-based margin calculation.

        Args:
            current_bar_data: DataFrame containing the current bar data
        """
        if not self.portfoliostate.open_positions:
            return
            
        current_bar = current_bar_data.iloc[-1]
        timestamp = current_bar_data['timestamp'].iloc[-1]

        position_margins = []
        total_unrealized_pnl = 0.0
        total_maintenance_margin_required = 0.0

        for position_id, position in self.portfoliostate.open_positions.items():
            if not position.is_margin:
                continue
                
            # Use worst-case price for conservative margin calculation
            if position.direction == OrderDirection.BUY:
                worst_case_price = current_bar['low']  # Longs hurt by low prices
            else:  # SHORT
                worst_case_price = current_bar['high']  # Shorts hurt by high prices
                
            # Calculate position metrics
            position_value = abs(position.quantity * worst_case_price)
            unrealized_pnl = position.get_unrealized_pnl(worst_case_price)
            
            # Calculate maintenance margin (typically half of initial margin)
            maintenance_margin_ratio = position.initial_margin_req * 0.5  # e.g., 25% if initial was 50%
            maintenance_margin_required = position_value * maintenance_margin_ratio
            
            position_margins.append({
                'position_id': position_id,
                'position': position,
                'position_value': position_value,
                'maintenance_margin_required': maintenance_margin_required,
                'liquidation_price': worst_case_price,
                'unrealized_pnl': unrealized_pnl
            })
            
            total_unrealized_pnl += unrealized_pnl
            total_maintenance_margin_required += maintenance_margin_required

        # Calculate account equity (cash + unrealized P&L)
        account_equity = self.portfoliostate.cash + total_unrealized_pnl
        
        # Check if we're below maintenance margin requirement
        if account_equity < total_maintenance_margin_required:
            margin_deficit = total_maintenance_margin_required - account_equity
            logger.warning(f"Margin call triggered: Account equity {account_equity:.2f} < Maintenance margin {total_maintenance_margin_required:.2f} (deficit: {margin_deficit:.2f})")
            
            self._liquidate_for_margin_call(position_margins, timestamp, account_equity, total_maintenance_margin_required)
        else:
            # Log margin health
            margin_cushion = account_equity - total_maintenance_margin_required
            logger.debug(f"Margin check passed: Equity {account_equity:.2f}, Maintenance {total_maintenance_margin_required:.2f}, Cushion {margin_cushion:.2f}")

    def _liquidate_for_margin_call(self, position_margins, timestamp, current_equity, current_maintenance_required):
        """
        Liquidate positions iteratively until maintenance margin requirements are met.
        
        Args:
            position_margins: List of dicts with position margin info
            timestamp: Current timestamp for trade records
            current_equity: Current account equity
            current_maintenance_required: Current maintenance margin requirement
        """
        if not position_margins:
            return

        # Sort positions by liquidation strategy (FIFO - oldest first)
        position_margins.sort(key=lambda x: x['position'].entry_time)

        liquidated_count = 0

        while True:
            # Recalculate maintenance margin requirement (excluding liquidated positions)
            remaining_maintenance_required = sum(
                pm['maintenance_margin_required'] for pm in position_margins 
                if pm['position_id'] in self.portfoliostate.open_positions
            )
            
            # Recalculate equity (cash + remaining unrealized P&L)
            remaining_unrealized_pnl = sum(
                pm['unrealized_pnl'] for pm in position_margins 
                if pm['position_id'] in self.portfoliostate.open_positions
            )
            current_equity = self.portfoliostate.cash + remaining_unrealized_pnl

            # Check if we now meet maintenance margin requirements
            if current_equity >= remaining_maintenance_required:
                if liquidated_count > 0:
                    margin_cushion = current_equity - remaining_maintenance_required
                    logger.info(f"Margin requirements satisfied after liquidating {liquidated_count} positions. "
                              f"Equity: {current_equity:.2f}, Required: {remaining_maintenance_required:.2f}")
                break

            # Find next position to liquidate
            position_to_liquidate = None
            for pm in position_margins:
                if pm['position_id'] in self.portfoliostate.open_positions:
                    position_to_liquidate = pm
                    break

            if position_to_liquidate is None:
                logger.error("No more positions to liquidate but margin requirements still not met!")
                break

            # Liquidate the position
            position_id = position_to_liquidate['position_id']
            liquidation_price = position_to_liquidate['liquidation_price']
            
            current_deficit = remaining_maintenance_required - current_equity
            logger.warning(f"Liquidating position {position_id} at {liquidation_price:.2f} (deficit: {current_deficit:.2f})")
            
            success = self.portfoliostate.liquidate_position(
                position_id=position_id,
                exit_price=liquidation_price,
                reason="Margin call",
                timestamp=timestamp
            )
            
            if success:
                liquidated_count += 1
            else:
                logger.error(f"Failed to liquidate position {position_id}")
                break

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