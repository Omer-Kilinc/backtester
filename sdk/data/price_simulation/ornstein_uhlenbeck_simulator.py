import pandas as pd
import numpy as np
from typing import Dict, Any

# TODO Ensure correctness of code

from sdk.configs.price_simulation.ornstein_uhlenbeck_simulator import OrnsteinUhlenbeckSimulatorConfig

class OrnsteinUhlenbeckSimulator:
    """Simulates an Ornstein-Uhlenbeck process with configurable parameters."""
    def __init__(self, config: OrnsteinUhlenbeckSimulatorConfig):
        self.config = config


    def simulate(self):
        if self.config.seed is not None:
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

        # Resample to OHLC format
        df_ohlc = df.resample(self.config.output_freq, on="timestamp").agg({
            "close": ["first", "max", "min", "last"]
        }).dropna()

        df_ohlc.columns = ["open", "high", "low", "close"]
        df_ohlc.reset_index(inplace=True)
        return df_ohlc


