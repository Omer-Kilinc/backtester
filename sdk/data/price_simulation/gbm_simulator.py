import pandas as pd
import numpy as np
import utils.logger as logger

# TODO Ensure correctness of code

from sdk.configs.price_simulation.gbm_simulator import GBMSimulatorConfig

class GeometricBrownianMotionSimulator:
    def __init__(self, config: GBMSimulatorConfig):
        self.config = config

    def simulate(self):
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # This is to determine the the base frequency for high-resolution simulation
        base_freq_seconds = pd.to_timedelta(self.config.output_freq).total_seconds() / self.config.precision_factor
        base_freq = f'{int(base_freq_seconds)}s'

        dates = pd.date_range(self.config.start_date, self.config.end_date, freq=base_freq)
        n = len(dates)
        dt = base_freq_seconds / (365 * 24 * 60 * 60)

        # Brownian motion increments
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=n)
        # Cumulative log returns
        log_returns = (self.config.mu - 0.5 * self.config.sigma**2) * dt + self.config.sigma * dW
        log_price = np.log(self.config.start_price) + np.cumsum(log_returns)
        prices = np.exp(log_price)

        df = pd.DataFrame({"timestamp": dates, "close": prices})

        # Resample to OHLC format
        df_ohlc = df.resample(self.config.output_freq, on="timestamp").agg({
            "close": ["first", "max", "min", "last"]
        }).dropna()

        df_ohlc.columns = ['open', 'high', 'low', 'close']
        df_ohlc.reset_index(inplace=True)
        logger.get_logger().info(f"Simulated {len(df_ohlc)} OHLC data points from {self.config.start_date} to {self.config.end_date} with frequency {self.config.output_freq}")
        return df_ohlc