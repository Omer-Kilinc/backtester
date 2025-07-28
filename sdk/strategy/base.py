import abc

class Strategy(abc.ABC):
    """
    Base class for all strategies, inheriting from this class will allow the user to define their own indicators to precompute.
    
    Attributes:
        precompute_indicators (list, optional): List of indicator names to precompute. Defaults to `[]`.
        indicator_funcs (dict): Dictionary of indicator functions and their precompute status.
    """
    def __init__(self):
        # Grab user-specified indicators to precompute from subclass
        precompute_list = getattr(self, 'precompute_indicators', [])

        self.indicator_funcs = {}
        for name, info in INDICATOR_REGISTRY.items():
            self.indicator_funcs[name] = {
                'func': info['func'],
                'precompute': name in precompute_list
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
    def on_bar(self, data: pd.DataFrame):
        """
        Called on each bar (i.e. after every new candle).
        Receives a pd.DataFrame including OHLCV and all computed indicator values
        up to and including the current bar.

        Args:
            data (pd.DataFrame): The data to process.

        Returns:
            List[TradeInstruction]: The trade instructions to execute.
            None: If no trade instruction is to be executed.
        """
        


    