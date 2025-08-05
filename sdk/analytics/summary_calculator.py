import pandas as pd
import numpy as np
from typing import Dict, List, Any
from logging import getLogger
from datetime import datetime

from sdk.backtester.portfoliostate import PortfolioState, ExecutedTrade
from sdk.configs.analytics.analytics import AnalyticsConfig

logger = getLogger(__name__)

# TODO: Implement Omega, Ulcer and K-Ratio

class SummaryCalculator:
    """Calculate end-of-backtest summary statistics"""
    
    def __init__(self, config: AnalyticsConfig, initial_capital: float):
        self.config = config
        self.initial_capital = initial_capital
        
    def calculate_all_metrics(self, data: pd.DataFrame, portfoliostate: PortfolioState) -> Dict[str, Any]:
        """Calculate all summary metrics"""
        logger.info("Calculating all summary metrics...")
        
        try:
            # Extract equity curve (portfolio values over time)
            equity_curve = data['portfolio_value'].dropna()
            
            if len(equity_curve) == 0:
                logger.warning("No portfolio value data found for metrics calculation")
                return {}
            
            # Calculate all metric categories
            metrics = {}
            metrics.update(self.calculate_returns_metrics(equity_curve))
            metrics.update(self.calculate_risk_metrics(equity_curve, data))
            metrics.update(self.calculate_trade_statistics(portfoliostate.executed_trades))
            metrics.update(self.calculate_position_statistics(portfoliostate))
            
            logger.info(f"Calculated {len(metrics)} summary metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating summary metrics: {e}")
            return {}
    
    def _calculate_annualized_return(self, final_value: float, initial_value: float, periods: int) -> float:
        """Centralized annualized return calculation"""
        if periods <= 0 or initial_value <= 0:
            return 0.0
        
        years = periods / 252  # Assuming 252 trading days per year
        if years <= 0:
            return 0.0
            
        return (final_value / initial_value) ** (1/years) - 1
    
    def calculate_returns_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate return-based metrics"""
        if len(equity_curve) < 2:
            return {}
            
        try:
            final_value = equity_curve.iloc[-1]
            initial_value = self.initial_capital
            
            # Basic returns
            total_return_pct = ((final_value - initial_value) / initial_value) * 100
            cumulative_return = final_value / initial_value - 1
            
            # Calculate daily returns for annualized metrics
            daily_returns = equity_curve.pct_change().dropna()
            
            if len(daily_returns) == 0:
                return {
                    'total_return_pct': total_return_pct,
                    'cumulative_return': cumulative_return
                }
            
            # Annualized return (using centralized calculation)
            trading_days = len(daily_returns)
            annualized_return = self._calculate_annualized_return(final_value, initial_value, trading_days)
            years = trading_days / 252
            
            # Best and worst single period returns
            best_day_return = daily_returns.max()
            worst_day_return = daily_returns.min()
            
            return {
                'total_return_pct': total_return_pct,
                'cumulative_return': cumulative_return,
                'annualized_return': annualized_return * 100,  # As percentage
                'best_day_return_pct': best_day_return * 100,
                'worst_day_return_pct': worst_day_return * 100,
                'trading_days': trading_days,
                'years_traded': years
            }
            
        except Exception as e:
            logger.error(f"Error calculating returns metrics: {e}")
            return {}
    
    def calculate_risk_metrics(self, equity_curve: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-based and risk-adjusted metrics"""
        if len(equity_curve) < 2:
            return {}
            
        try:
            # Calculate daily returns
            daily_returns = equity_curve.pct_change().dropna()
            
            if len(daily_returns) == 0:
                return {}
            
            # Volatility (annualized)
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Maximum drawdown from tracked data
            if 'drawdown_pct' in data.columns:
                max_drawdown_pct = data['drawdown_pct'].max()
                max_drawdown_abs = data['drawdown'].max()
            else:
                # Calculate manually if not tracked
                running_max = equity_curve.expanding().max()
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown_pct = abs(drawdown.min()) * 100
                max_drawdown_abs = (running_max - equity_curve).max()
            
            # Risk-adjusted returns
            excess_returns = daily_returns - (self.config.risk_free_rate / 252)  # Daily risk-free rate
            
            # Sharpe Ratio (annualized)
            sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Sortino Ratio (using downside deviation only)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            
            # Calmar Ratio (Annualized Return / Max Drawdown) - use consistent annualized return
            annualized_return_for_calmar = self._calculate_annualized_return(
                equity_curve.iloc[-1], 
                self.initial_capital, 
                len(daily_returns)
            )
            calmar_ratio = (annualized_return_for_calmar / (max_drawdown_pct/100)) if max_drawdown_pct > 0 else 0
            
            # Value at Risk (VaR) - This it the maximum expected loss at confidence level
            var_confidence_pct = self.config.var_confidence_level * 100
            var_daily = -np.percentile(daily_returns, 100 - var_confidence_pct) 
            var_annual = var_daily * np.sqrt(252)
            
            # Maximum consecutive losing days
            losing_streaks = self._calculate_consecutive_periods(daily_returns < 0)
            max_consecutive_losses = max(losing_streaks) if losing_streaks else 0
            
            return {
                'volatility_annualized': volatility * 100,  # As percentage
                'max_drawdown_pct': max_drawdown_pct,
                'max_drawdown_abs': max_drawdown_abs,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'var_daily_pct': var_daily * 100,
                'var_annual_pct': var_annual * 100,
                'max_consecutive_losing_days': max_consecutive_losses
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def calculate_trade_statistics(self, executed_trades: List[ExecutedTrade]) -> Dict[str, float]:
        """Calculate trade-based statistics"""
        if not executed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0
            }
        
        try:
            # Filter for exit trades only (these have realized P&L)
            exit_trades = [trade for trade in executed_trades if not trade.is_entry and trade.realized_pnl is not None]
            
            if not exit_trades:
                return {
                    'total_trades': len(executed_trades),
                    'entry_trades': len([t for t in executed_trades if t.is_entry]),
                    'exit_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'expectancy': 0
                }
            
            # Calculate P&L statistics
            pnl_values = [trade.realized_pnl for trade in exit_trades]
            winning_trades = [pnl for pnl in pnl_values if pnl > 0]
            losing_trades = [pnl for pnl in pnl_values if pnl < 0]
            
            # Basic statistics
            total_trades = len(exit_trades)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            win_rate = (winning_count / total_trades) * 100 if total_trades > 0 else 0
            
            # Profit factor (gross profits / gross losses)
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            # Expectancy (average profit per trade)
            expectancy = sum(pnl_values) / total_trades if total_trades > 0 else 0
            
            # Average win vs average loss
            avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Largest win and loss
            largest_win = max(winning_trades) if winning_trades else 0
            largest_loss = min(losing_trades) if losing_trades else 0
            
            # Commission impact
            total_commission = sum(trade.commission for trade in executed_trades)
            total_fees = sum(trade.fees for trade in executed_trades)
            
            return {
                'total_trades': total_trades,
                'entry_trades': len([t for t in executed_trades if t.is_entry]),
                'exit_trades': total_trades,
                'winning_trades': winning_count,
                'losing_trades': losing_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'total_commission': total_commission,
                'total_fees': total_fees
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return {}
    
    def calculate_position_statistics(self, portfoliostate: PortfolioState) -> Dict[str, Any]:
        """Calculate position-based statistics (all positions liquidated at end of backtest)"""
        try:
            # Final portfolio state (after liquidation)
            total_cash = portfoliostate.cash
            
            # Historical positions (all positions are now closed)
            total_positions = len(portfoliostate.closed_positions_history)
            
            # Verify liquidation was successful
            remaining_open_positions = len(portfoliostate.open_positions)
            if remaining_open_positions > 0:
                logger.warning(f"Warning: {remaining_open_positions} positions still open after liquidation")
            
            return {
                'total_positions': total_positions,
                'final_cash': total_cash,
                'remaining_open_positions': remaining_open_positions,  # Should be 0
                # Note: No unrealized_pnl, used_margin, short_proceeds needed since all positions closed
            }
            
        except Exception as e:
            logger.error(f"Error calculating position statistics: {e}")
            return {}
    
    def _calculate_consecutive_periods(self, condition_series: pd.Series) -> List[int]:
        """Calculate consecutive periods where condition is True"""
        streaks = []
        current_streak = 0
        
        for value in condition_series:
            if value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0
        
        # Don't forget the last streak if it ends at the series end
        if current_streak > 0:
            streaks.append(current_streak)
            
        return streaks