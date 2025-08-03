from sdk.backtester.portfoliostate import PortfolioState
from sdk.configs.backtester.backtester import BacktesterConfig
from sdk.strategy.base import Strategy
from sdk.data.data_pipeline import DataPipelineConfig, DataPipeline
from sdk.configs.backtester.tradeinstruction import OrderDirection, OrderType, FailureReason
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
        
        # Order management
        self.queued_market_orders = {}  # Orders to execute at next bar open

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
            
            # Execute any queued market orders at bar open
            self._execute_queued_market_orders(current_bar_data)
            
            # Process intrabar events (margin, exits, pending order triggers)
            self._process_intrabar_events(current_bar_data)
            
            # Call strategy to get new trade instructions (at bar close)
            trade_instructions = self.strategy.on_bar(current_bar_data)
            
            # Process new trade instructions from strategy
            if trade_instructions:
                self._process_trade_instructions(trade_instructions, current_bar_data)
        
        logger.info("Backtest completed.")
        self.strategy.on_teardown()

    def _process_intrabar_events(self, current_bar_data):
        """Process all intrabar events in priority order"""
        logger.debug(f"Processing intrabar events for bar with OHLC: {current_bar_data.iloc[-1]['open']:.2f}, {current_bar_data.iloc[-1]['high']:.2f}, {current_bar_data.iloc[-1]['low']:.2f}, {current_bar_data.iloc[-1]['close']:.2f}")

        # Conservative approach:
        # 1. Checking for margin requirements
        self._check_margin_requirements(current_bar_data)
        
        # 2. Process Stop Losses
        self._process_stop_losses(current_bar_data)

        # 3. Process Take Profits
        self._process_take_profits(current_bar_data)
        
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
        if position_id not in self.portfoliostate.open_positions:
            logger.error(f"Cannot close {position_id} - position not found")
            return False
        
        position = self.portfoliostate.open_positions[position_id]
        
        # Calculate commission on exit trade
        trade_value = position.quantity * exit_price
        commission = self.config.calculate_commission(trade_value)
        
        # Calculate realized P&L before commission
        realized_pnl = position.get_unrealized_pnl(exit_price)
        
        # Update cash based on position direction
        if position.direction == OrderDirection.BUY:
            # Closing long position: sell shares, add proceeds minus commission
            net_proceeds = (position.quantity * exit_price) - commission
            self.portfoliostate.cash += net_proceeds
        else:  # OrderDirection.SELL (short position)
            # Closing short position: buy back shares plus commission
            buyback_cost = (position.quantity * exit_price) + commission
            self.portfoliostate.cash -= buyback_cost
            
            # Unlock short sale proceeds
            if position.metadata and isinstance(position.metadata, dict):
                short_proceeds = position.metadata.get('short_proceeds_locked', 0.0)
                if short_proceeds > 0:
                    self.portfoliostate.unlock_short_proceeds(short_proceeds)
        
        # Unlock margin that was locked for this position
        if position.is_margin and position.initial_margin_locked:
            self.portfoliostate.unlock_margin(position.initial_margin_locked)
        
        # Create trade record with commission
        from sdk.backtester.portfoliostate import ExecutedTrade
        trade = ExecutedTrade(
            symbol=position.symbol,
            direction=OrderDirection.SELL if position.direction == OrderDirection.BUY else OrderDirection.BUY,
            quantity=position.quantity,
            price=exit_price,
            timestamp=timestamp,
            is_entry=False,
            realized_pnl=realized_pnl - commission,  # Subtract commission from P&L
            commission=commission,
            fees=0.0,
            position_id=position_id
        )
        
        # Update records
        self.portfoliostate.executed_trades.append(trade)
        self.portfoliostate.closed_positions_history.append(position)
        del self.portfoliostate.open_positions[position_id]
        
        logger.info(f"Closed position {position.symbol} (ID: {position_id}) at {exit_price:.2f} due to {reason}. "
                   f"P&L: {realized_pnl:.2f}, Commission: {commission:.2f}, Net P&L: {realized_pnl - commission:.2f}")
        return True

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