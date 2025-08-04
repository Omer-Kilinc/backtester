import abc
import pandas as pd
from typing import List, Optional, Union
from sdk.strategy.registry import INDICATOR_REGISTRY
from sdk.configs.backtester.tradeinstruction import TradeInstruction

class Strategy(abc.ABC):
    """
    Base class for all strategies, inheriting from this class will allow the user to define their own indicators to precompute.
    
    Attributes:
        precompute_indicators (list, optional): List of indicator names to precompute. Defaults to `[]`.
        indicator_funcs (dict): Dictionary of indicator functions and their precompute status.
    """
    def __init__(self):
        precompute_list = getattr(self, 'precompute_indicators', [])
        
        self.indicator_funcs = {}
        for name, info in INDICATOR_REGISTRY.items():
            is_vectorized = info.get('vectorized', False)
            user_wants_precompute = name in precompute_list
            
            self.indicator_funcs[name] = {
                'func': info['func'],
                'vectorized': is_vectorized,
                'precompute': is_vectorized or user_wants_precompute
            }
    
    def on_init(self):
        """
        Called once at the start of the backtest.
        Can be used to set internal state or print a startup message.
        """
        pass

    def on_teardown(self):
        """
        Called once at the end of the backtest.
        Use this to clean up, log information, or finalize anything.
        """
        pass
    
    @abc.abstractmethod
    def on_bar(self, data: pd.DataFrame) -> Optional[Union[TradeInstruction, List[TradeInstruction]]]:
        """
        Called on each bar (i.e. after every new candle).
        Receives a pd.DataFrame including OHLCV and all computed indicator values
        up to and including the current bar.

        Args:
            data (pd.DataFrame): The data to process, including all historical data up to current bar.

        Returns:
            TradeInstruction: A single trade instruction to execute.
            List[TradeInstruction]: Multiple trade instructions to execute.
            None: If no trade instruction is to be executed.
        """
        pass