import numpy as np
import pandas as pd
from typing import Dict, Optional
from pydantic import ValidationError

from utils.logger import get_logger
from sdk.configs.data.processing.noise_generator import NoiseGeneratorConfig

logger = get_logger(__name__)


class NoiseGenerator:
    """Injects Gaussian noise and simulates gaps in OHLC data."""

    def __init__(self, config: NoiseGeneratorConfig):
        self.config = config
        self.df = config.df.copy()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def inject_noise_with_gaps(self) -> pd.DataFrame:
        logger.info("Starting noise and gap injection...")
        required_cols = self.config.required_columns
        ohlc_data = self.df[required_cols].values

        # Generate noise
        noise = np.random.normal(1, [self.config.scale[col] for col in required_cols], size=ohlc_data.shape)
        ohlc_data *= noise

        # Generate gaps
        self._apply_gaps(ohlc_data)

        # Ensure candlestick integrity
        self._ensure_candlestick_integrity(ohlc_data)

        self.df[required_cols] = ohlc_data
        logger.info(f"Injected noise and gaps into {len(self.df)} rows.")
        return self.df

    def _apply_gaps(self, ohlc_data):
        gap_mask = np.random.uniform(0, 1, len(self.df)) < self.config.gap_chance
        gap_indices = np.where(gap_mask)[0]

        if len(gap_indices) > 0:
            directions = np.random.choice([1, -1], len(gap_indices))
            gap_pcts = np.random.uniform(self.config.gap_magnitude["min"], self.config.gap_magnitude["max"], len(gap_indices))
            gap_factors = 1 + (directions * gap_pcts)
            ohlc_data[gap_indices] *= gap_factors[:, np.newaxis]
            logger.info(f"Applied {len(gap_indices)} gaps.")

    def _ensure_candlestick_integrity(self, ohlc_data):
        low_col_idx = self.config.required_columns.index('low')
        high_col_idx = self.config.required_columns.index('high')

        # Ensure low is the minimum and high is the maximum
        ohlc_data[:, low_col_idx] = np.min(ohlc_data, axis=1)
        ohlc_data[:, high_col_idx] = np.max(ohlc_data, axis=1)


def inject_noise(
    *,
    df: pd.DataFrame,
    scale: Dict[str, float] = {"open": 0.001, "high": 0.001, "low": 0.001, "close": 0.001},
    gap_chance: float = 0.01,
    gap_magnitude: Dict[str, float] = {"min": 0.005, "max": 0.02},
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    User-facing function to inject noise and gaps into OHLC data.

    Args:
        df: The input DataFrame with OHLC data.
        scale: The scale of Gaussian noise for each column.
        gap_chance: The probability of a gap occurring.
        gap_magnitude: The min and max magnitude of gaps.
        seed: An optional random seed for reproducibility.

    Returns:
        A new DataFrame with noise and gaps injected.

    Raises:
        pydantic.ValidationError: If any of the parameters fail validation.
    """
    try:
        config = NoiseGeneratorConfig(
            df=df,
            scale=scale,
            gap_chance=gap_chance,
            gap_magnitude=gap_magnitude,
            seed=seed
        )
        generator = NoiseGenerator(config)
        return generator.inject_noise_with_gaps()
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during noise injection: {e}", exc_info=True)
        raise