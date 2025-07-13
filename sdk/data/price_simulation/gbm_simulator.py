import pandas as pd
import numpy as np
from typing import Optional
from pydantic import ValidationError

from utils.logger import get_logger
from sdk.configs.data.price_simulation.gbm_simulator import GBMSimulatorConfig

logger = get_logger(__name__)


class GeometricBrownianMotionSimulator:
    def __init__(self, config: GBMSimulatorConfig):
        self.config = config

    def simulate(self) -> pd.DataFrame:
        if self.config.seed is not None:
            logger.info(f"Using random seed: {self.config.seed}")
            np.random.seed(self.config.seed)

        base_freq_seconds = pd.to_timedelta(self.config.output_freq).total_seconds() / self.config.precision_factor
        base_freq = f'{int(base_freq_seconds)}s'

        dates = pd.date_range(self.config.start_date, self.config.end_date, freq=base_freq)
        n = len(dates)
        dt = base_freq_seconds / (365 * 24 * 60 * 60)

        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=n)
        log_returns = (self.config.mu - 0.5 * self.config.sigma**2) * dt + self.config.sigma * dW
        log_price = np.log(self.config.start_price) + np.cumsum(log_returns)
        prices = np.exp(log_price)

        df = pd.DataFrame({"timestamp": dates, "close": prices})

        df_ohlc = df.resample(self.config.output_freq, on="timestamp").agg(
            open=('close', 'first'),
            high=('close', 'max'),
            low=('close', 'min'),
            close=('close', 'last')
        ).dropna()

        df_ohlc.reset_index(inplace=True)
        logger.info(f"Simulated {len(df_ohlc)} OHLC data points from {self.config.start_date} to {self.config.end_date}")
        return df_ohlc


def gbm_simulator(
    *,
    start_price: float,
    mu: float,
    sigma: float,
    start_date: str,
    end_date: str,
    output_freq: str = "1D",
    precision_factor: int = 24,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    User-facing function to simulate price paths using Geometric Brownian Motion (GBM).

    This function validates inputs, initializes the simulator, runs the simulation,
    and returns the generated OHLC data as a pandas DataFrame.

    Args:
        start_price: The initial price of the asset.
        mu: The drift (expected return) of the asset.
        sigma: The volatility of the asset.
        start_date: The start date for the simulation period (e.g., '2023-01-01').
        end_date: The end date for the simulation period (e.g., '2023-12-31').
        output_freq: The frequency of the output data (e.g., '1D' for daily).
        precision_factor: The number of high-resolution steps within each output frequency period.
        seed: An optional random seed for reproducibility.

    Returns:
        A pandas DataFrame with the simulated OHLC data.

    Raises:
        pydantic.ValidationError: If any of the parameters fail validation.
    """
    try:
        config = GBMSimulatorConfig(
            start_price=start_price,
            mu=mu,
            sigma=sigma,
            start_date=start_date,
            end_date=end_date,
            output_freq=output_freq,
            precision_factor=precision_factor,
            seed=seed
        )
        simulator = GeometricBrownianMotionSimulator(config)
        return simulator.simulate()
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during GBM simulation: {e}", exc_info=True)
        raise