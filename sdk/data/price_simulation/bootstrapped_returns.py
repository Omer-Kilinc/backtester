import pandas as pd
import numpy as np
from utils.logger import get_logger
from sdk.configs.data.price_simulation.bootstrapped_returns import BootstrapReturnSimulatorConfig

# TODO Ensure correctness of code
logger = get_logger(__name__)

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
    
    def _generate_brownian_path(self, open_price, close_price, num_points=100):
        """Generate a Brownian bridge path between the open and close prices."""

        # TODO: Add volatility scaling based on return magnitude
        # TODO: Implement time-dependent volatility (U-shaped intraday pattern)

        logger.info("Generating Brownian bridge path")

        # Work in log space to prevent negative prices
        log_open = np.log(open_price)
        log_close = np.log(close_price)
        
        # Generate Random Walk in log space
        path = [log_open]
        for i in range(1, num_points):
            step = np.random.normal(0, np.sqrt(1 / num_points))
            path.append(path[-1] + step)

        T = num_points - 1
        W_T = path[-1]
        adjusted_path = [
            W_t - (t / T) * (W_T - log_close) 
            for t, W_t in enumerate(path)
        ]

        # Convert back to price space
        price_path = [np.exp(log_price) for log_price in adjusted_path]

        high = max(price_path)
        low = min(price_path)

        return high, low
        
    def simulate(self):
        """Run the bootstrap return simulation."""

        if self.config.seed is not None:
            logger.info(f"Using seed: {self.config.seed}")
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
                high, low = self._generate_brownian_path(open_price, close_price)
            else:
                high = max(open_price, close_price)
                low = min(open_price, close_price)

            ohlc_data.append([date, open_price, high, low, close_price])
            current_price = close_price

        return pd.DataFrame(ohlc_data, columns=['date', 'open', 'high', 'low', 'close'])


def bootstrap_return_simulator(
    *,
    cleaned_price_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_freq: str = "1D",
    use_path_simulation: bool = True,
    inject_gaps: bool = False,
    gap_volatility: float = 0.01,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    User-facing function to simulate financial returns using bootstrapping.

    This function validates inputs, initializes the simulator, runs the simulation,
    and returns the generated OHLC data as a pandas DataFrame.

    Args:
        cleaned_price_df: DataFrame with historical price data, must include a 'close' column.
        start_date: The start date for the simulation period (e.g., '2023-01-01').
        end_date: The end date for the simulation period (e.g., '2023-12-31').
        output_freq: The frequency of the output data (e.g., '1D' for daily).
        use_path_simulation: If True, generates a Brownian bridge for intraday OHLC.
        inject_gaps: If True, introduces price gaps between simulation periods.
        gap_volatility: The volatility of the price gaps, if enabled.
        seed: An optional random seed for reproducibility.

    Returns:
        A pandas DataFrame with the simulated OHLC data.

    Raises:
        pydantic.ValidationError: If any of the parameters fail validation.
    """
    try:
        config = BootstrapReturnSimulatorConfig(
            cleaned_price_df=cleaned_price_df,
            start_date=start_date,
            end_date=end_date,
            output_freq=output_freq,
            use_path_simulation=use_path_simulation,
            inject_gaps=inject_gaps,
            gap_volatility=gap_volatility,
            seed=seed
        )
        simulator = BootstrapReturnSimulator(config)
        return simulator.simulate()
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during simulation: {e}", exc_info=True)
        raise