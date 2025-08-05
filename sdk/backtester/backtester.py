from sdk.backtester.portfoliostate import PortfolioState
from sdk.configs.backtester.backtester import BacktesterConfig
from sdk.strategy.base import Strategy
from sdk.data.data_pipeline import DataPipelineConfig, DataPipeline
from sdk.configs.backtester.tradeinstruction import OrderDirection, OrderType, FailureReason
from sdk.configs.analytics.analytics import AnalyticsConfig
from sdk.analytics.performance_analyzer import PerformanceAnalyzer
from typing import TypeVar, Union, List, Dict, Any
from logging import getLogger
from sdk.strategy.registry import INDICATOR_REGISTRY
import pandas as pd
import numpy as np

logger = getLogger(__name__)

T = TypeVar('T', bound=Strategy)

class Backtester:
    """
    Backtester class for running backtests

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
        data: pd.DataFrame = None,
        analytics_config: Optional[AnalyticsConfig] = None
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

        self.analytics = None
        if analytics_config:
            # Assume strategy has symbol attribute - adjust if needed
            symbol = getattr(strategy, 'symbol', 'UNKNOWN')
            self.analytics = PerformanceAnalyzer(
                config=analytics_config,
                symbol=symbol,
                initial_capital=config.initial_capital
            )
        
        self.queued_market_orders = {}  # Orders to execute at next bar open
        
        self.SUPPORTED_ACTIONS = {
            'close': self._handle_close_action,
            'update_stop': self._handle_update_stop_action,
            'update_take_profit': self._handle_update_take_profit_action,
            'update_trailing_stop': self._handle_update_trailing_stop_action,
            'cancel_orders': self._handle_cancel_orders_action,
            'place_order': self._handle_place_order_action
        }

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

    def execute_backtest(self) -> Dict[str, Any]:
        """
        Execute the complete backtest workflow and return results.
        
        The backtest process includes:
        1. Strategy initialization
        2. Data preparation (if needed)
        3. Indicator precomputation
        4. Sequential bar processing:
            - Market order execution
            - Intrabar event processing
            - Live indicator calculation
            - Analytics tracking (if enabled)
            - Trade instruction processing
        5. Final metrics calculation (if analytics enabled)
        6. Strategy cleanup
    
        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'data' (pd.DataFrame): 
                    The complete dataset including:
                    - OHLCV prices
                    - Calculated indicators
                    - Analytics columns (if analytics enabled)
                - 'portfoliostate' (PortfolioState): 
                    Final portfolio state including:
                    - Current positions
                    - Cash balance
                    - Portfolio metrics
                - 'final_metrics' (Optional[StrategyMetrics]): 
                    Performance metrics (None if analytics disabled), typically including:
                    - Sharpe ratio
                    - Max drawdown
                    - Win rate
                    - Other strategy metrics
                - 'trade_history' (List[Trade]):
                    Chronological list of all executed trades during backtest
                    Each Trade object contains:
                    - Entry/exit prices and timestamps
                    - Position size
                    - PnL
                    - Metadata
                - 'analytics' (PerformanceAnalyzer):
                    Analytics object instance containing:
                    - Final metrics
                    - Performance summary
        """
        logger.info("Executing backtest...")
        self.strategy.on_init()
        
        if self.data is None:
            self.prepare_data()

        self.precompute_indicators(progress_log_freq=self.config.progress_log_freq)

        if self.analytics:
            self.analytics.initialize_data_columns(self.data)
            # Auto-register decorated metrics from strategy
            self.analytics.register_strategy_metrics(self.strategy)
        
        for i in range(len(self.data)):
            current_bar_data = self.data.iloc[:i+1]
            
            # Execute any queued market orders at bar open
            self._execute_queued_market_orders(current_bar_data)
            
            # Process intrabar events (margin, exits, pending order triggers)
            self._process_intrabar_events(current_bar_data)

            # Compute live indicators for current bar before calling strategy
            self._compute_live_indicators(current_bar_data, i)

            if self.analytics:
                self.analytics.track_bar(self.portfoliostate, current_bar_data, i)
            
            # Call strategy to get new trade instructions (at bar close)
            trade_instructions = self.strategy.on_bar(current_bar_data)
            
            # Process new trade instructions from strategy
            if trade_instructions:
                self._process_trade_instructions(trade_instructions, current_bar_data)
        
        if self.portfoliostate.open_positions and len(self.data) > 0:
            final_close_price = self.data.iloc[-1]['close']
            final_timestamp = self.data.index[-1] if hasattr(self.data.index[-1], 'to_pydatetime') else pd.Timestamp.now()
            
            logger.info(f"Liquidating {len(self.portfoliostate.open_positions)} remaining positions...")
            
            # Get list of position IDs to avoid dict modification during iteration
            positions_to_close = list(self.portfoliostate.open_positions.keys())
            
            for position_id in positions_to_close:
                try:
                    success = self.portfoliostate.liquidate_position(
                        position_id=position_id,
                        liquidation_price=final_close_price,
                        reason="End of backtest liquidation",
                        timestamp=final_timestamp,
                        commission_rate=self.config.commission_rate
                    )
                    
                    if success:
                        logger.debug(f"Liquidated position {position_id} at {final_close_price:.2f}")
                    else:
                        logger.warning(f"Failed to liquidate position {position_id}")
                        
                except Exception as e:
                    logger.error(f"Error liquidating position {position_id}: {e}")
        
        logger.info(f"End-of-backtest liquidation complete. Remaining positions: {len(self.portfoliostate.open_positions)}")

        final_metrics = None
        if self.analytics:
            final_metrics = self.analytics.compute_final_metrics(self.data, self.portfoliostate)
            logger.info("Analytics calculations completed.")
            
            # Log performance summary
            summary = self.analytics.get_performance_summary(final_metrics)
        logger.info(f"\n{summary}")

        logger.info("Backtest completed.")
        self.strategy.on_teardown()

        return {
            'data': self.data,  # DataFrame with OHLCV + indicators + analytics columns
            'portfoliostate': self.portfoliostate,
            'final_metrics': final_metrics,
            'trade_history': self.portfoliostate.executed_trades,
            'analytics': self.analytics
        }

    def _process_intrabar_events(self, current_bar_data):
        """Process all intrabar events in priority order"""
        logger.debug(f"Processing intrabar events for bar with OHLC: {current_bar_data.iloc[-1]['open']:.2f}, {current_bar_data.iloc[-1]['high']:.2f}, {current_bar_data.iloc[-1]['low']:.2f}, {current_bar_data.iloc[-1]['close']:.2f}")

        # Conservative approach - order matters:
        # 1. Check margin requirements first (most critical)
        self._check_margin_requirements(current_bar_data)
        
        # 2. Process Stop Losses (risk management)
        self._process_stop_losses(current_bar_data)

        # 3. Process Take Profits (profit taking)
        self._process_take_profits(current_bar_data)
        
        # 4. Process Custom Exit Conditions (user-defined logic)
        self._process_custom_exit_conditions(current_bar_data)
        
        # 5. Process pending orders (limit/stop orders)
        self._process_pending_orders(current_bar_data)

    def _process_custom_exit_conditions(self, current_bar_data: pd.DataFrame):
        """
        Process custom exit conditions for all positions.
        Supports multiple actions per position with robust error handling.
        
        Args:
            current_bar_data: All historical data up to current bar
        """
        if not self.portfoliostate.open_positions:
            return
            
        current_bar_index = len(current_bar_data) - 1
        timestamp = current_bar_data['timestamp'].iloc[-1]
        
        # Create a copy of positions to iterate over (since we might modify the dict)
        positions_to_check = list(self.portfoliostate.open_positions.items())
        
        for position_id, position in positions_to_check:
            # Skip if position doesn't have exit condition or not time to check
            if not position.should_check_exit_condition(current_bar_index):
                continue
            
            try:
                # Create read-only copies for safety (trust user but provide immutable-like access)
                readonly_data = current_bar_data.copy()
                readonly_position = position
                readonly_portfolio = self.portfoliostate
                
                # Call user's exit condition function
                result = position.exit_condition(readonly_data, readonly_position, readonly_portfolio)
                
                # Parse and validate the result
                actions = self._parse_exit_condition_result(result)
                
                if actions:
                    logger.debug(f"Position {position_id} exit condition returned {len(actions)} actions")
                    
                    # Execute all valid actions (skip failed ones, continue with others)
                    for action in actions:
                        try:
                            self._execute_exit_action(position_id, action, current_bar_data, timestamp)
                        except Exception as e:
                            logger.error(f"Failed to execute action {action} for position {position_id}: {e}")
                            # Continue with next action
                
                # Update last check bar (only if position still exists)
                if position_id in self.portfoliostate.open_positions:
                    self.portfoliostate.open_positions[position_id].last_exit_check_bar = current_bar_index
                    
            except Exception as e:
                logger.error(f"Error in exit condition for position {position_id}: {e}")
                # Continue backtesting with other positions

    def _parse_exit_condition_result(self, result: Union[bool, float, dict, List[dict]]) -> List[dict]:
        """
        Parse the result from an exit condition function into a list of actions.
        
        Args:
            result: Return value from exit condition function
            
        Returns:
            List[dict]: List of validated action dictionaries
        """
        if result is None or result is False:
            return []
        
        actions = []
        
        if result is True:
            # Simple close entire position
            actions.append({'action': 'close', 'percentage': 1.0, 'reason': 'Exit condition triggered'})
            
        elif isinstance(result, (int, float)):
            # Percentage close
            if 0 < result <= 1:
                actions.append({'action': 'close', 'percentage': float(result), 'reason': 'Partial exit condition'})
            else:
                logger.error(f"Invalid percentage {result} - must be between 0 and 1")
                
        elif isinstance(result, dict):
            # Single action dictionary
            if self._validate_action(result):
                actions.append(result)
                
        elif isinstance(result, list):
            # Multiple actions
            for action in result:
                if isinstance(action, dict) and self._validate_action(action):
                    actions.append(action)
                else:
                    logger.error(f"Invalid action in list: {action}")
        else:
            logger.error(f"Unsupported exit condition result type: {type(result)}")
            
        return actions

    def _validate_action(self, action: dict) -> bool:
        """
        Validate an action dictionary.
        
        Args:
            action: Action dictionary to validate
            
        Returns:
            bool: True if action is valid
        """
        if not isinstance(action, dict):
            return False
            
        action_type = action.get('action')
        if action_type not in self.SUPPORTED_ACTIONS:
            logger.error(f"Unsupported action type: {action_type}")
            return False
        
        # Validate action-specific parameters
        if action_type == 'close':
            percentage = action.get('percentage', 1.0)
            if not isinstance(percentage, (int, float)) or not 0 < percentage <= 1:
                logger.error(f"Invalid close percentage: {percentage}")
                return False
                
        elif action_type in ['update_stop', 'update_take_profit', 'update_trailing_stop']:
            price_key = {
                'update_stop': 'new_stop',
                'update_take_profit': 'new_tp',
                'update_trailing_stop': 'new_trailing'
            }[action_type]
            
            if price_key not in action:
                logger.error(f"Missing {price_key} for {action_type}")
                return False
                
        elif action_type == 'place_order':
            if 'instruction' not in action:
                logger.error("Missing 'instruction' for place_order action")
                return False
        
        return True

    def _execute_exit_action(self, position_id: str, action: dict, current_bar_data: pd.DataFrame, timestamp):
        """
        Execute a single exit action.
        
        Args:
            position_id: Position ID to act on
            action: Action dictionary
            current_bar_data: Current bar data
            timestamp: Current timestamp
        """
        if position_id not in self.portfoliostate.open_positions:
            logger.warning(f"Position {position_id} not found for action {action}")
            return
            
        action_type = action['action']
        action_handler = self.SUPPORTED_ACTIONS.get(action_type)
        
        if action_handler:
            action_handler(position_id, action, current_bar_data, timestamp)
        else:
            logger.error(f"No handler for action type: {action_type}")

    def _handle_close_action(self, position_id: str, action: dict, current_bar_data: pd.DataFrame, timestamp):
        """Handle close position action"""
        percentage = action.get('percentage', 1.0)
        reason = action.get('reason', 'Custom exit condition')
        
        # Use current close price for exit
        current_price = current_bar_data.iloc[-1]['close']
        
        success = self.portfoliostate.close_position_partially(
            position_id=position_id,
            percentage=percentage,
            exit_price=current_price,
            reason=reason,
            timestamp=timestamp,
            commission_rate=self.config.commission_rate
        )
        
        if success:
            logger.info(f"Executed close action: {percentage*100:.1f}% of position {position_id}")
        else:
            logger.error(f"Failed to close position {position_id}")

    def _handle_update_stop_action(self, position_id: str, action: dict, current_bar_data: pd.DataFrame, timestamp):
        """Handle update stop loss action"""
        new_stop = action.get('new_stop')
        position = self.portfoliostate.open_positions[position_id]
        
        # Validate new stop makes sense
        current_price = current_bar_data.iloc[-1]['close']
        
        if isinstance(new_stop, str) and new_stop.endswith('%'):
            # Convert percentage to absolute price
            percent = float(new_stop.rstrip('%')) / 100
            if position.direction == OrderDirection.BUY:
                new_stop = position.entry_price * (1 - percent)
            else:
                new_stop = position.entry_price * (1 + percent)
        
        # Basic validation
        if position.direction == OrderDirection.BUY and new_stop >= current_price:
            logger.warning(f"Invalid stop loss {new_stop} for long position at {current_price}")
            return
        elif position.direction == OrderDirection.SELL and new_stop <= current_price:
            logger.warning(f"Invalid stop loss {new_stop} for short position at {current_price}")
            return
        
        # Update the stop loss
        old_stop = position.exit_stop_loss
        position.exit_stop_loss = new_stop
        
        logger.info(f"Updated stop loss for position {position_id}: {old_stop} → {new_stop}")

    def _handle_update_take_profit_action(self, position_id: str, action: dict, current_bar_data: pd.DataFrame, timestamp):
        """Handle update take profit action"""
        new_tp = action.get('new_tp')
        position = self.portfoliostate.open_positions[position_id]
        
        # Validate new take profit makes sense
        current_price = current_bar_data.iloc[-1]['close']
        
        if isinstance(new_tp, str) and new_tp.endswith('%'):
            # Convert percentage to absolute price
            percent = float(new_tp.rstrip('%')) / 100
            if position.direction == OrderDirection.BUY:
                new_tp = position.entry_price * (1 + percent)
            else:
                new_tp = position.entry_price * (1 - percent)
        
        # Basic validation
        if position.direction == OrderDirection.BUY and new_tp <= current_price:
            logger.warning(f"Invalid take profit {new_tp} for long position at {current_price}")
            return
        elif position.direction == OrderDirection.SELL and new_tp >= current_price:
            logger.warning(f"Invalid take profit {new_tp} for short position at {current_price}")
            return
        
        # Update the take profit
        old_tp = position.take_profit
        position.take_profit = new_tp
        
        logger.info(f"Updated take profit for position {position_id}: {old_tp} → {new_tp}")

    def _handle_update_trailing_stop_action(self, position_id: str, action: dict, current_bar_data: pd.DataFrame, timestamp):
        """Handle update trailing stop action"""
        new_trailing = action.get('new_trailing')
        position = self.portfoliostate.open_positions[position_id]
        
        # Update the trailing stop
        old_trailing = position.exit_trailing_stop
        position.exit_trailing_stop = new_trailing
        
        logger.info(f"Updated trailing stop for position {position_id}: {old_trailing} → {new_trailing}")

    def _handle_cancel_orders_action(self, position_id: str, action: dict, current_bar_data: pd.DataFrame, timestamp):
        """Handle cancel orders action"""
        # Find and cancel any pending orders related to this position
        orders_to_cancel = []
        
        for order_id, order in self.portfoliostate.pending_orders.items():
            # Match orders by metadata or other criteria
            if order.metadata and order.metadata.get('position_id') == position_id:
                orders_to_cancel.append(order_id)
        
        for order_id in orders_to_cancel:
            del self.portfoliostate.pending_orders[order_id]
            logger.info(f"Cancelled order {order_id} for position {position_id}")

    def _handle_place_order_action(self, position_id: str, action: dict, current_bar_data: pd.DataFrame, timestamp):
        """Handle place new order action"""
        instruction = action.get('instruction')
        
        if not instruction:
            logger.error("No instruction provided for place_order action")
            return
        
        # Add position reference to metadata
        if not instruction.metadata:
            instruction.metadata = {}
        instruction.metadata['related_position_id'] = position_id
        
        # Process the new trade instruction
        try:
            self._process_trade_instructions([instruction], current_bar_data)
            logger.info(f"Placed new order for position {position_id}")
        except Exception as e:
            logger.error(f"Failed to place order for position {position_id}: {e}")

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

        # Calculate account equity (total collateral + unrealized P&L)
        account_equity = self.portfoliostate.total_collateral + total_unrealized_pnl
        
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
            
            # Recalculate equity (total collateral + remaining unrealized P&L)
            remaining_unrealized_pnl = sum(
                pm['unrealized_pnl'] for pm in position_margins 
                if pm['position_id'] in self.portfoliostate.open_positions
            )
            current_equity = self.portfoliostate.total_collateral + remaining_unrealized_pnl

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

    def _process_stop_losses(self, current_bar_data):
        """
        Process stop loss orders for all open positions.
        Uses conservative approach - assumes stops hit at worst possible price.
        
        TODO: Implement other execution models:
        - Stop hits at exact stop price (less conservative)
        - HLOC sequence execution (Level 2)
        - Randomized execution within bar range
        
        Args:
            current_bar_data: All historical data up to current bar
        """
        if not self.portfoliostate.open_positions:
            return
            
        current_bar = current_bar_data.iloc[-1]
        timestamp = current_bar_data['timestamp'].iloc[-1]
        
        positions_to_close = []
        
        for position_id, position in self.portfoliostate.open_positions.items():
            if position.exit_stop_loss is None:
                continue
                
            # Calculate stop loss price
            stop_loss_price = self._get_stop_loss_price(position)
            if stop_loss_price is None:
                continue
                
            # Check if stop loss was triggered during this bar
            stop_triggered = False
            exit_price = None
            
            if position.direction == OrderDirection.BUY:
                # Long position: stop loss below current price
                if current_bar['low'] <= stop_loss_price:
                    stop_triggered = True
                    # Conservative: assume we got stopped at worst possible price
                    exit_price = min(stop_loss_price, current_bar['low'])
                    
            else:  # OrderDirection.SELL (short position)
                # Short position: stop loss above current price  
                if current_bar['high'] >= stop_loss_price:
                    stop_triggered = True
                    # Conservative: assume we got stopped at worst possible price
                    exit_price = max(stop_loss_price, current_bar['high'])
            
            if stop_triggered:
                positions_to_close.append((position_id, exit_price, "Stop loss"))
                logger.info(f"Stop loss triggered for {position.symbol} (ID: {position_id}) at {exit_price:.2f} (stop: {stop_loss_price:.2f})")
        
        # Execute stop loss exits
        for position_id, exit_price, reason in positions_to_close:
            success = self.portfoliostate.liquidate_position(
                position_id=position_id,
                exit_price=exit_price,
                reason=reason,
                timestamp=timestamp
            )
            if not success:
                logger.error(f"Failed to execute stop loss for position {position_id}")

    def _process_take_profits(self, current_bar_data):
        """
        Process take profit orders for all open positions.
        Assumes take profits hit at the take profit price (less conservative than stops).
        
        TODO: Add config option for commission timing:
        - 'entry_only': Commission only on position entry
        - 'exit_only': Commission only on position exit  
        - 'both': Commission on both entry and exit (current default)
        
        Args:
            current_bar_data: All historical data up to current bar
        """
        if not self.portfoliostate.open_positions:
            return
            
        current_bar = current_bar_data.iloc[-1]
        timestamp = current_bar_data['timestamp'].iloc[-1]
        
        positions_to_close = []
        
        for position_id, position in self.portfoliostate.open_positions.items():
            if position.take_profit is None:
                continue
                
            # Calculate take profit price
            take_profit_price = self._get_take_profit_price(position)
            if take_profit_price is None:
                continue
                
            # Check if take profit was triggered during this bar
            tp_triggered = False
            exit_price = None
            
            if position.direction == OrderDirection.BUY:
                # Long position: take profit above current price
                if current_bar['high'] >= take_profit_price:
                    tp_triggered = True
                    # Execute at take profit price with slippage (less favorable)
                    slippage_factor = 1 - self.config.slippage_model  # Sell lower due to slippage
                    exit_price = take_profit_price * slippage_factor
                    
            else:  # OrderDirection.SELL (short position)
                # Short position: take profit below current price
                if current_bar['low'] <= take_profit_price:
                    tp_triggered = True
                    # Execute at take profit price with slippage (less favorable)
                    slippage_factor = 1 + self.config.slippage_model  # Buy higher due to slippage
                    exit_price = take_profit_price * slippage_factor
            
            if tp_triggered:
                positions_to_close.append((position_id, exit_price, "Take profit"))
                logger.info(f"Take profit triggered for {position.symbol} (ID: {position_id}) at {exit_price:.2f} (target: {take_profit_price:.2f})")
        
        # Execute take profit exits with commission
        for position_id, exit_price, reason in positions_to_close:
            success = self._close_position_with_commission(
                position_id=position_id,
                exit_price=exit_price,
                reason=reason,
                timestamp=timestamp
            )
            if not success:
                logger.error(f"Failed to execute take profit for position {position_id}")

    def _get_take_profit_price(self, position) -> Optional[float]:
        """
        Extract take profit price from position, handling both absolute and percentage values.
        
        Args:
            position: Position object with take_profit
            
        Returns:
            Optional[float]: Take profit price, or None if not set/invalid
        """
        if position.take_profit is None:
            return None
            
        try:
            if isinstance(position.take_profit, str) and position.take_profit.endswith('%'):
                # Percentage-based take profit
                percent = float(position.take_profit.rstrip('%')) / 100
                if position.direction == OrderDirection.BUY:
                    # Long: take profit above entry price
                    return position.entry_price * (1 + percent)
                else:
                    # Short: take profit below entry price
                    return position.entry_price * (1 - percent)
            else:
                # Absolute price take profit
                return float(position.take_profit)
                
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid take profit value for position {position.symbol}: {position.take_profit} - {e}")
            return None

    def _close_position_with_commission(self, position_id: str, exit_price: float, reason: str, timestamp) -> bool:
        """
        Close a position with commission applied to the exit trade.
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price before commission/slippage
            reason: Reason for closing
            timestamp: Exit timestamp
            
        Returns:
            bool: True if successful, False if position not found
        """
        return self.portfoliostate.close_position_partially(
            position_id=position_id,
            percentage=1.0,
            exit_price=exit_price,
            reason=reason,
            timestamp=timestamp,
            commission_rate=self.config.commission_rate
        )

    def _get_stop_loss_price(self, position) -> Optional[float]:
        """
        Extract stop loss price from position, handling both absolute and percentage values.
        
        Args:
            position: Position object with exit_stop_loss
            
        Returns:
            Optional[float]: Stop loss price, or None if not set/invalid
        """
        if position.exit_stop_loss is None:
            return None
            
        try:
            if isinstance(position.exit_stop_loss, str) and position.exit_stop_loss.endswith('%'):
                # Percentage-based stop loss
                percent = float(position.exit_stop_loss.rstrip('%')) / 100
                if position.direction == OrderDirection.BUY:
                    # Long: stop loss below entry price
                    return position.entry_price * (1 - percent)
                else:
                    # Short: stop loss above entry price
                    return position.entry_price * (1 + percent)
            else:
                # Absolute price stop loss
                return float(position.exit_stop_loss)
                
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid stop loss value for position {position.symbol}: {position.exit_stop_loss} - {e}")
            return None

    def _execute_queued_market_orders(self, current_bar_data):
        """
        Execute all queued market orders at bar open.
        
        Args:
            current_bar_data: All historical data up to current bar
        """
        if not self.queued_market_orders:
            return
            
        current_bar = current_bar_data.iloc[-1]
        execution_price = current_bar['open']
        timestamp = current_bar_data['timestamp'].iloc[-1]
        
        orders_to_remove = []
        executed_count = 0
        
        for order_id, instruction in self.queued_market_orders.items():
            try:
                # Convert amount to quantity if needed
                quantity = self._convert_amount_to_quantity(instruction, execution_price)
                
                # Validate the trade before execution
                is_valid, error_msg = self.portfoliostate.validate_trade(
                    direction=instruction.direction,
                    quantity=quantity,
                    price=execution_price,
                    is_margin=instruction.is_margin,
                    leverage=instruction.leverage,
                    margin_requirement=instruction.margin_requirement,
                    commission_rate=self.config.commission_rate,
                    allow_short_selling=self.config.allow_short_selling,
                    require_margin_for_shorts=self.config.require_margin_for_shorts,
                    max_leverage=self.config.max_leverage
                )
                
                if not is_valid:
                    logger.warning(f"Market order {order_id} validation failed: {error_msg}")
                    self._handle_order_failure(instruction, error_msg, current_bar_data)
                    orders_to_remove.append(order_id)
                    continue
                
                # Execute the trade
                if instruction.direction == OrderDirection.BUY:
                    position_id = self.portfoliostate.execute_buy(
                        symbol=self.config.symbol,
                        quantity=quantity,
                        price=execution_price,
                        timestamp=timestamp,
                        is_margin=instruction.is_margin,
                        leverage=instruction.leverage,
                        margin_requirement=instruction.margin_requirement,
                        take_profit=instruction.take_profit,
                        exit_stop_loss=instruction.exit_stop_loss,
                        exit_trailing_stop=instruction.exit_trailing_stop,
                        exit_condition=instruction.exit_condition,
                        exit_condition_frequency=getattr(instruction, 'exit_condition_frequency', 1),
                        metadata=instruction.metadata,
                        commission_rate=self.config.commission_rate,
                        slippage_rate=self.config.slippage_model
                    )
                else:  # SELL
                    position_id = self.portfoliostate.execute_sell(
                        symbol=self.config.symbol,
                        quantity=quantity,
                        price=execution_price,
                        timestamp=timestamp,
                        is_margin=instruction.is_margin,
                        leverage=instruction.leverage,
                        margin_requirement=instruction.margin_requirement,
                        take_profit=instruction.take_profit,
                        exit_stop_loss=instruction.exit_stop_loss,
                        exit_trailing_stop=instruction.exit_trailing_stop,
                        exit_condition=instruction.exit_condition,
                        exit_condition_frequency=getattr(instruction, 'exit_condition_frequency', 1),
                        metadata=instruction.metadata,
                        commission_rate=self.config.commission_rate,
                        slippage_rate=self.config.slippage_model,
                        allow_short_selling=self.config.allow_short_selling,
                        require_margin_for_shorts=self.config.require_margin_for_shorts
                    )
                
                logger.info(f"Executed market order {order_id} → position {position_id}")
                executed_count += 1
                orders_to_remove.append(order_id)
                
            except Exception as e:
                logger.error(f"Failed to execute market order {order_id}: {e}")
                self._handle_order_failure(instruction, str(e), current_bar_data)
                orders_to_remove.append(order_id)
        
        # Remove executed/failed orders
        for order_id in orders_to_remove:
            del self.queued_market_orders[order_id]
            
        if executed_count > 0:
            logger.info(f"Executed {executed_count} market orders at {execution_price:.2f}")

    def _process_trade_instructions(self, trade_instructions, current_bar_data):
        """
        Process trade instructions from strategy and route them appropriately.
        
        Args:
            trade_instructions: Single TradeInstruction or list of TradeInstructions
            current_bar_data: All historical data up to current bar
        """
        # Convert single instruction to list
        if not isinstance(trade_instructions, list):
            trade_instructions = [trade_instructions]
        
        for instruction in trade_instructions:
            try:
                if instruction.order_type == OrderType.MARKET:
                    self._queue_market_order(instruction)
                else:
                    # LIMIT, STOP, STOP_LIMIT orders go to pending
                    self._add_to_pending_orders(instruction)
                    
            except Exception as e:
                logger.error(f"Failed to process trade instruction: {e}")
                self._handle_order_failure(instruction, str(e), current_bar_data)


    def _queue_market_order(self, instruction):
        """
        Queue a market order for execution at next bar open.
        
        Args:
            instruction: TradeInstruction with order_type=MARKET
        """
        order_id = f"market_{self.portfoliostate.order_counter}"
        self.portfoliostate.order_counter += 1
        
        self.queued_market_orders[order_id] = instruction
        logger.debug(f"Queued market order {order_id} for next bar open")


    def _add_to_pending_orders(self, instruction):
        """
        Add a limit/stop order to pending orders.
        
        Args:
            instruction: TradeInstruction with order_type=LIMIT/STOP/STOP_LIMIT
        """
        order_id = f"pending_{self.portfoliostate.order_counter}"
        self.portfoliostate.order_counter += 1
        
        current_bar_index = len(self.data) - 1 if self.data is not None else 0
        self.portfoliostate.order_placement_bars[order_id] = current_bar_index
        
        self.portfoliostate.pending_orders[order_id] = instruction
        logger.debug(f"Added {instruction.order_type.value} order {order_id} to pending orders")

    def _convert_amount_to_quantity(self, instruction, price):
        """
        Convert TradeInstruction amount to quantity.
        
        Args:
            instruction: TradeInstruction with either amount or quantity
            price: Current price for conversion
            
        Returns:
            float: Quantity of shares
            
        Raises:
            ValueError: If neither amount nor quantity specified
        """
        if instruction.quantity is not None:
            return instruction.quantity
        elif instruction.amount is not None:
            return instruction.amount / price
        else:
            raise ValueError("TradeInstruction must specify either amount or quantity")

    def _handle_order_failure(self, instruction, error_msg, current_bar_data):
        """
        Handle failed order execution based on configuration and callbacks.
        
        Args:
            instruction: Failed TradeInstruction
            error_msg: Error message describing the failure
            current_bar_data: Current bar data for callback
        """
        # Call user's failure callback if provided
        if instruction.on_fail:
            try:
                # Determine failure reason
                if "Insufficient" in error_msg:
                    if "margin" in error_msg.lower():
                        failure_reason = FailureReason.InsufficientMargin
                    else:
                        failure_reason = FailureReason.InsufficientFunds
                else:
                    failure_reason = FailureReason.InsufficientFunds  # Default
                
                instruction.on_fail(current_bar_data, instruction, failure_reason)
                logger.debug(f"Called on_fail callback for failed order: {error_msg}")
                
            except Exception as e:
                logger.error(f"Error in on_fail callback: {e}")
        
        # Apply configured failure action
        if self.config.failed_order_action == 'keep':
            # For market orders, we can't really "keep" them since they execute immediately
            # This would be more relevant for pending orders that fail validation
            logger.info(f"Order failed but keeping per config: {error_msg}")
        elif self.config.failed_order_action == 'remove':
            logger.info(f"Removing failed order: {error_msg}")
        # 'callback' action just calls the callback above

    def precompute_indicators(self, progress_log_freq: float = 10.0):
        """
        Precompute indicators with adjustable progress logging.
        
        Args:
            progress_log_freq: Percentage (0-100) for progress updates. 
                        0 = off, 10 = every 10% (default), 100 = every indicator.
        """
        logger.info("Precomputing indicators...")

        # Use strategy's indicator configuration instead of global registry
        indicators = {
            name: info for name, info in self.strategy.indicator_funcs.items() 
            if info.get('precompute', False)
        }
        
        if not indicators:
            logger.info("No indicators to precompute")
            return
        
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
    

    def _process_pending_orders(self, current_bar_data: pd.DataFrame):
        """
        Process all pending orders at bar close.
        Handle expiration, triggering, and execution in FIFO order.
        
        Args:
            current_bar_data: All historical data up to current bar
        """
        if not self.portfoliostate.pending_orders:
            return
            
        # Step 1: Remove expired orders first
        self._expire_pending_orders(current_bar_data)
        
        if not self.portfoliostate.pending_orders:
            return
            
        # Step 2: Get current bar info
        current_bar = current_bar_data.iloc[-1]
        timestamp = current_bar_data['timestamp'].iloc[-1]
        
        # Step 3: Process orders in FIFO order (by order_id which includes counter)
        orders_to_remove = []
        orders_to_modify = []  # For STOP_LIMIT conversions
        
        # Sort by order ID to maintain FIFO (order_counter ensures chronological order)
        for order_id in sorted(self.portfoliostate.pending_orders.keys()):
            instruction = self.portfoliostate.pending_orders[order_id]
            
            # Check if order should trigger
            triggered, execution_price = self._check_order_trigger(instruction, current_bar)
            
            if triggered:
                if instruction.order_type == OrderType.STOP_LIMIT:
                    # Convert to LIMIT order, don't execute yet
                    new_instruction = self._convert_stop_limit_to_limit(instruction)
                    orders_to_modify.append((order_id, new_instruction))
                    logger.info(f"STOP_LIMIT order {order_id} stop triggered, converted to LIMIT order")
                else:
                    # Try to execute the order
                    success = self._execute_pending_order(instruction, execution_price, timestamp, current_bar_data)
                    if success:
                        orders_to_remove.append(order_id)
                        logger.info(f"Executed pending order {order_id} at {execution_price:.2f}")
                    else:
                        logger.debug(f"Pending order {order_id} triggered but failed validation - keeping in queue")
                    # If failed validation, keep in pending orders
        
        # Step 4: Apply modifications and removals
        for order_id in orders_to_remove:
            del self.portfoliostate.pending_orders[order_id]
            if order_id in self.portfoliostate.order_placement_bars:
                del self.portfoliostate.order_placement_bars[order_id]
            
        for order_id, new_instruction in orders_to_modify:
            self.portfoliostate.pending_orders[order_id] = new_instruction
            # Keep same placement bar for expiration tracking

    def _expire_pending_orders(self, current_bar_data: pd.DataFrame):
        """
        Remove expired orders from pending queue.
        
        Args:
            current_bar_data: All historical data up to current bar
        """
        current_bar_index = len(current_bar_data) - 1
        orders_to_remove = []
        
        for order_id, instruction in self.portfoliostate.pending_orders.items():
            if instruction.time_in_force == TimeInForce.GTD:
                placement_bar = self.portfoliostate.order_placement_bars.get(order_id, 0)
                bars_since_placement = current_bar_index - placement_bar
                
                if bars_since_placement >= instruction.good_till_date:
                    orders_to_remove.append(order_id)
                    logger.info(f"Order {order_id} expired after {bars_since_placement} bars (GTD: {instruction.good_till_date})")
            
            # TODO: Implement DAY order expiration
            # elif instruction.time_in_force == TimeInForce.DAY:
            #     # Check if we've crossed a trading day boundary
            #     pass
        
        # Remove expired orders
        for order_id in orders_to_remove:
            del self.portfoliostate.pending_orders[order_id]
            if order_id in self.portfoliostate.order_placement_bars:
                del self.portfoliostate.order_placement_bars[order_id]

    def _check_order_trigger(self, instruction, current_bar) -> tuple[bool, float]:
        """
        Determine if a pending order should trigger based on current bar OHLC.
        
        Args:
            instruction: TradeInstruction to check
            current_bar: Current bar data with OHLC
            
        Returns:
            tuple[bool, float]: (triggered, execution_price)
        """
        triggered = False
        execution_price = 0.0
        
        if instruction.order_type == OrderType.LIMIT:
            if instruction.direction == OrderDirection.BUY:
                # LIMIT BUY: Trigger when price goes at or below limit
                if current_bar['low'] <= instruction.price:
                    triggered = True
                    execution_price = self._calculate_execution_price(instruction, current_bar)
            else:  # SELL
                # LIMIT SELL: Trigger when price goes at or above limit  
                if current_bar['high'] >= instruction.price:
                    triggered = True
                    execution_price = self._calculate_execution_price(instruction, current_bar)
                    
        elif instruction.order_type == OrderType.STOP:
            if instruction.direction == OrderDirection.BUY:
                # STOP BUY: Trigger when price goes at or above stop
                if current_bar['high'] >= instruction.stop_price:
                    triggered = True
                    execution_price = self._calculate_execution_price(instruction, current_bar)
            else:  # SELL
                # STOP SELL: Trigger when price goes at or below stop
                if current_bar['low'] <= instruction.stop_price:
                    triggered = True
                    execution_price = self._calculate_execution_price(instruction, current_bar)
                    
        elif instruction.order_type == OrderType.STOP_LIMIT:
            # Check only the stop condition - limit will be checked after conversion
            if instruction.direction == OrderDirection.BUY:
                # STOP_LIMIT BUY: Stop triggers when price goes at or above stop
                if current_bar['high'] >= instruction.stop_price:
                    triggered = True
                    # Execution price not relevant here - will become LIMIT order
            else:  # SELL
                # STOP_LIMIT SELL: Stop triggers when price goes at or below stop
                if current_bar['low'] <= instruction.stop_price:
                    triggered = True
                    # Execution price not relevant here - will become LIMIT order
        
        return triggered, execution_price

    def _calculate_execution_price(self, instruction, current_bar) -> float:
        """
        Calculate actual execution price, handling gaps appropriately.
        
        Args:
            instruction: TradeInstruction with price/stop_price
            current_bar: Current bar OHLC data
            
        Returns:
            float: Actual execution price
        """
        if instruction.order_type == OrderType.LIMIT:
            target_price = instruction.price
            
            if instruction.direction == OrderDirection.BUY:
                # LIMIT BUY: Execute at limit price or better (lower)
                # Handle gap down: if bar low is better than limit, execute at open
                if current_bar['open'] <= target_price:
                    return current_bar['open']  # Gap down - better execution
                else:
                    return target_price  # Normal execution at limit
            else:  # SELL
                # LIMIT SELL: Execute at limit price or better (higher)  
                # Handle gap up: if bar high is better than limit, execute at open
                if current_bar['open'] >= target_price:
                    return current_bar['open']  # Gap up - better execution
                else:
                    return target_price  # Normal execution at limit
                    
        elif instruction.order_type == OrderType.STOP:
            # STOP orders execute at market price (with slippage)
            market_price = current_bar['open']  # Conservative: assume triggered at open
            
            if instruction.direction == OrderDirection.BUY:
                # Apply slippage: buy higher
                return market_price * (1 + self.config.slippage_model)
            else:  # SELL
                # Apply slippage: sell lower
                return market_price * (1 - self.config.slippage_model)
        
        # Fallback
        return current_bar['open']

    def _convert_stop_limit_to_limit(self, instruction):
        """
        Convert a triggered STOP_LIMIT order to a regular LIMIT order.
        
        Args:
            instruction: STOP_LIMIT TradeInstruction
            
        Returns:
            TradeInstruction: New LIMIT order
        """
        # Create new instruction as LIMIT order
        from copy import deepcopy
        new_instruction = deepcopy(instruction)
        new_instruction.order_type = OrderType.LIMIT
        # Keep the limit price, remove stop_price (though it won't be used)
        
        return new_instruction

    def _execute_pending_order(self, instruction, execution_price: float, timestamp, current_bar_data: pd.DataFrame) -> bool:
        """
        Execute a triggered pending order with validation.
        
        Args:
            instruction: TradeInstruction to execute
            execution_price: Price to execute at
            timestamp: Execution timestamp
            current_bar_data: Current bar data for callbacks
            
        Returns:
            bool: True if executed successfully, False if validation failed
        """
        try:
            # Convert amount to quantity using execution price (current trigger price)
            quantity = self._convert_amount_to_quantity(instruction, execution_price)
            
            # Validate the trade before execution
            is_valid, error_msg = self.portfoliostate.validate_trade(
                direction=instruction.direction,
                quantity=quantity,
                price=execution_price,
                is_margin=instruction.is_margin,
                leverage=instruction.leverage,
                margin_requirement=instruction.margin_requirement,
                commission_rate=self.config.commission_rate,
                allow_short_selling=self.config.allow_short_selling,
                require_margin_for_shorts=self.config.require_margin_for_shorts,
                max_leverage=self.config.max_leverage
            )
            
            if not is_valid:
                logger.debug(f"Pending order validation failed: {error_msg}")
                return False  # Keep order in pending queue
            
            # Execute the trade
            if instruction.direction == OrderDirection.BUY:
                position_id = self.portfoliostate.execute_buy(
                    symbol=self.config.symbol,
                    quantity=quantity,
                    price=execution_price,
                    timestamp=timestamp,
                    is_margin=instruction.is_margin,
                    leverage=instruction.leverage,
                    margin_requirement=instruction.margin_requirement,
                    take_profit=instruction.take_profit,
                    exit_stop_loss=instruction.exit_stop_loss,
                    exit_trailing_stop=instruction.exit_trailing_stop,
                    exit_condition=instruction.exit_condition,
                    exit_condition_frequency=getattr(instruction, 'exit_condition_frequency', 1),
                    metadata=instruction.metadata,
                    commission_rate=self.config.commission_rate,
                    slippage_rate=0.0  # Already applied in execution_price calculation
                )
            else:  # SELL
                position_id = self.portfoliostate.execute_sell(
                    symbol=self.config.symbol,
                    quantity=quantity,
                    price=execution_price,
                    timestamp=timestamp,
                    is_margin=instruction.is_margin,
                    leverage=instruction.leverage,
                    margin_requirement=instruction.margin_requirement,
                    take_profit=instruction.take_profit,
                    exit_stop_loss=instruction.exit_stop_loss,
                    exit_trailing_stop=instruction.exit_trailing_stop,
                    exit_condition=instruction.exit_condition,
                    exit_condition_frequency=getattr(instruction, 'exit_condition_frequency', 1),
                    metadata=instruction.metadata,
                    commission_rate=self.config.commission_rate,
                    slippage_rate=0.0,  # Already applied in execution_price calculation
                    allow_short_selling=self.config.allow_short_selling,
                    require_margin_for_shorts=self.config.require_margin_for_shorts
                )
            
            logger.info(f"Executed pending order: {instruction.direction.value} {quantity} {self.config.symbol} at {execution_price:.2f} → position {position_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute pending order: {e}")
            self._handle_order_failure(instruction, str(e), current_bar_data)
            return False
    
    def _compute_live_indicators(self, current_bar_data: pd.DataFrame, bar_index: int):
        """
        Compute non-precomputed indicators for the current bar.
        
        Args:
            current_bar_data: All historical data up to current bar (view of self.data)
            bar_index: Index of the current bar in the dataset
        """
        live_indicators = {
            name: info for name, info in self.strategy.indicator_funcs.items() 
            if not info.get('precompute', False)
        }
        
        if not live_indicators:
            return  # No live indicators to compute
        
        for name in live_indicators.keys():
            if name not in current_bar_data.columns:
                current_bar_data[name] = np.nan
        
        for name, info in live_indicators.items():
            try:
                if info.get('vectorized', False):
                    logger.warning(f"Vectorized indicator '{name}' should be precomputed, computing anyway")
                    indicator_values = info['func'](current_bar_data)
                    current_bar_data[name] = indicator_values
                else:
                    indicator_value = info['func'](current_bar_data)
                    
                    col_idx = current_bar_data.columns.get_loc(name)
                    current_bar_data.iat[bar_index, col_idx] = indicator_value
                    
                    logger.debug(f"Computed live indicator '{name}' for bar {bar_index}: {indicator_value}")
                    
            except Exception as e:
                logger.error(f"Failed to compute live indicator '{name}' for bar {bar_index}: {e}")
                col_idx = current_bar_data.columns.get_loc(name)
                current_bar_data.iat[bar_index, col_idx] = np.nan