import pandas as pd
import numpy as np
from typing import Optional
from pydantic import ValidationError

from utils.logger import get_logger
from sdk.configs.data.price_simulation.ornstein_uhlenbeck_simulator import OrnsteinUhlenbeckSimulatorConfig

logger = get_logger(__name__)


class OrnsteinUhlenbeckSimulator:
    """Simulates an Ornstein-Uhlenbeck process with configurable parameters."""
    def __init__(self, config: OrnsteinUhlenbeckSimulatorConfig):
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

        x = np.zeros(n)
        x[0] = self.config.x0

        for i in range(1, n):
            dx = self.config.theta * (self.config.mu - x[i-1]) * dt + self.config.sigma * np.sqrt(dt) * np.random.normal()
            x[i] = x[i-1] + dx

        df = pd.DataFrame({"timestamp": dates, "close": x})

        df_ohlc = df.resample(self.config.output_freq, on="timestamp").agg(
            open=('close', 'first'),
            high=('close', 'max'),
            low=('close', 'min'),
            close=('close', 'last')
        ).dropna()

        df_ohlc.reset_index(inplace=True)
        logger.info(f"Simulated {len(df_ohlc)} OHLC data points from {self.config.start_date} to {self.config.end_date}")
        return df_ohlc


def ornstein_uhlenbeck_simulator(
    *,
    start_date: str,
    end_date: str,
    x0: float = 100.0,
    mu: float = 100.0,
    theta: float = 0.1,
    sigma: float = 1.0,
    output_freq: str = "1D",
    precision_factor: int = 24,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    User-facing function to simulate a mean-reverting process using the Ornstein-Uhlenbeck model.

    This function validates inputs, initializes the simulator, runs the simulation,
    and returns the generated OHLC data as a pandas DataFrame.

    Args:
        start_date: The start date for the simulation period (e.g., '2023-01-01').
        end_date: The end date for the simulation period (e.g., '2023-12-31').
        x0: The initial value of the process.
        mu: The long-term mean of the process.
        theta: The speed of reversion to the mean.
        sigma: The volatility of the process.
        output_freq: The frequency of the output data (e.g., '1D' for daily).
        precision_factor: The number of high-resolution steps within each output frequency period.
        seed: An optional random seed for reproducibility.

    Returns:
        A pandas DataFrame with the simulated OHLC data.

    Raises:
        pydantic.ValidationError: If any of the parameters fail validation.
    """
    try:
        config = OrnsteinUhlenbeckSimulatorConfig(
            x0=x0,
            mu=mu,
            theta=theta,
            sigma=sigma,
            start_date=start_date,
            end_date=end_date,
            output_freq=output_freq,
            precision_factor=precision_factor,
            seed=seed
        )
        simulator = OrnsteinUhlenbeckSimulator(config)
        return simulator.simulate()
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during Ornstein-Uhlenbeck simulation: {e}", exc_info=True)
        raise


