import pandas as pd
import numpy as np
import utils.logger as logger

# TODO Ensure correctness of code

from sdk.configs.price_simulation.bootstrapped_returns import BootstrapReturnSimulatorConfig

class BootstrapReturnSimulator:
    """Simulates financial returns using bootstrapping from historical data."""
    def __init__(self, config: BootstrapReturnSimulatorConfig):
        self.config = config
        self.returns = None

    def _get_pct_returns(self):
        """Calculate percentage returns from the cleaned price DataFrame."""
        self.returns = self.config.cleaned_price_df['close'].pct_change().dropna().values
    
    def _sample_return(self):
        """Randomly sample a return from the historical returns."""
        if self.returns is None:
            self._get_pct_returns()
        return np.random.choice(self.returns)
    
    def _generate_brownian_path(self, open_price, close_price):
        """Generate a Brownian bridge path between the open and close prices."""
        # TODO Implement the Brownian bridge path generation logic
        pass
        
    def simulate(self):
        """Run the bootstrap return simulation."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq=self.config.output_freq)
        ohlc_data = []

        # Start at some initial price
        current_price = self.config.cleaned_price_df['close'].iloc[0]

        for i, date in enumerate(dates):
            # Determine open price
            if self.config.inject_gaps and i > 0:  # Don't gap the first day
                gap = np.random.normal(0, self.config.gap_volatility)
                open_price = current_price * (1 + gap)
            else:
                open_price = current_price

            r = self._sample_return()
            close_price = open_price * (1 + r)

            if self.config.use_path_simulation:
                path = self._generate_brownian_path(open_price, close_price)
                if path is not None:  
                    high = max(path)
                    low = min(path)
                else:
                    high = max(open_price, close_price)
                    low = min(open_price, close_price)
            else:
                high = max(open_price, close_price)
                low = min(open_price, close_price)

            ohlc_data.append([date, open_price, high, low, close_price])
            current_price = close_price

        return pd.DataFrame(ohlc_data, columns=['date', 'open', 'high', 'low', 'close'])