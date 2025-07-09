import numpy as np
import pandas as pd
from typing import Dict, Any
from utils.logger import get_logger

# TODO Ensure correctness of code

from sdk.configs.noise_generator import NoiseGeneratorConfig

def inject_noise_with_gaps(df, config: NoiseGeneratorConfig):
    """Injects Gaussian noise into OHLC data and simulates gaps.

    Args:  
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close'].
        config (Dict[str, Any]): Configuration dictionary with keys:
            - scale (Dict[str, float]): Standard deviations for noise in each OHLC field.
            - gap_chance (float): Probability of a gap occurring.
            - gap_magnitude (Dict[str, float]): Min and max percentage for gap size.
    Returns:
        pd.DataFrame: DataFrame with injected noise and simulated gaps.
    """
    logger = get_logger()
    
    df_noisy = df.copy()
    
    ohlc_fields = ["open", "high", "low", "close"]
    
    for idx in df_noisy.index:
        # Step 1: Apply Gaussian noise to each OHLC field
        for field in ohlc_fields:
            noise = 1 + np.random.normal(mean=0, scale=config.scale[field])
            df_noisy.loc[idx, field] *= noise
        
        # Ensure candlestick Integrity
        df_noisy.loc[idx, "low"] = min(
            df_noisy.loc[idx, "open"], 
            df_noisy.loc[idx, "close"], 
            df_noisy.loc[idx, "low"]
        )
        df_noisy.loc[idx, "high"] = max(
            df_noisy.loc[idx, "open"], 
            df_noisy.loc[idx, "close"], 
            df_noisy.loc[idx, "high"]
        )
        
        # Occasionally simulate gap ups or downs
        if np.random.uniform(0, 1) < config.gap_chance:
            direction = np.random.choice(["up", "down"])
            gap_pct = np.random.uniform(
                config.gap_magnitude["min"], 
                config.gap_magnitude["max"]
            )
            
            # Apply the gap by shifting all OHLC values proportionally
            gap_factor = 1 + gap_pct if direction == "up" else 1 - gap_pct
            for field in ohlc_fields:
                df_noisy.loc[idx, field] *= gap_factor
            
            logger.info(f"Applied {direction} gap of {gap_pct:.3f} at index {idx}")
    
    logger.info(f"Injected noise and gaps into {len(df_noisy)} rows")
    return df_noisy