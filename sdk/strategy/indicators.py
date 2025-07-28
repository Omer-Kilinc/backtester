import pandas as pd
import numpy as np
from typing import Dict
from sdk.strategy.base import indicator

# TODO: Write docstrings? Don't think it's necessary, because the functions are only called by the backtester, not the user. The user accesses them via the dataframe.


@indicator(name='sma', precompute=False, vectorized=True)
def _sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Internal helper function for Simple Moving Average
    
    Args:
        df: The input DataFrame with all market data.
        period: The period to use for the moving average.
    
    Returns:
        A pandas Series with the SMA values (with NaN for first period-1 values)
    """
    return df['close'].rolling(window=period).mean()



@indicator(name='ema', precompute=False, vectorized=True)
def _ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Internal helper function for Exponential Moving Average (EMA)
    
    Args:
        df: Input DataFrame with market data.
        period: EMA period (default: 20).
    
    Returns:
        pd.Series: EMA values (NaN for first `period-1` entries).
    """
    return df['close'].ewm(span=period, adjust=False).mean()

@indicator(name='rsi', precompute=False, vectorized=False)
def _rsi(df: pd.DataFrame, period: int = 14) -> float:
    """
    Internal helper function for Relative Strength Index

    Args:
        df: The input DataFrame with all market data.
        period: The period to use for the RSI.
    """
    if len(df) < period + 1:
        return np.nan
    if 'close' not in df.columns:
        return np.nan
        
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

@indicator(name='rsi', precompute=False, vectorized=True)
def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Vectorized Relative Strength Index (RSI)

    Args:
        df: Input DataFrame with market data.
        period: RSI period (default: 14).

    Returns:
        pd.Series: RSI values (NaN for first `period` entries).
    """
    if len(df) < period + 1:
        return pd.Series(np.nan, index=df.index)  # Return NaNs if insufficient data
    
    if 'close' not in df.columns:
        return pd.Series(np.nan, index=df.index)  # Handle missing 'close' column
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Exponential Moving Averages for gains & losses
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    # Relative Strength & RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi  # Returns full Series (vectorized)

@indicator(name='atr', precompute=False, vectorized=True)
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Vectorized Average True Range (ATR)

    Args:
        df: Input DataFrame with market data (must contain 'high', 'low', 'close').
        period: ATR period (default: 14).

    Returns:
        pd.Series: ATR values (NaN for first `period` entries).
    """
    if len(df) < period + 1:
        return pd.Series(np.nan, index=df.index)  # Return NaNs if insufficient data
    
    # Check if required columns exist
    required_cols = {'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)  # Handle missing columns
    
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    # Calculate True Range (TR)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # Calculate ATR (using SMA of TR)
    atr = tr.rolling(window=period).mean()
    
    return atr 

@indicator(name='bollinger_bands', precompute=False, vectorized=True)
def _bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    ma_type: Literal['sma', 'ema'] = 'sma'  # New parameter
) -> Dict[str, Union[pd.Series, float]]:
    """
    Internal helper function for Bollinger Bands with SMA/EMA option
    
    Args:
        df: Input DataFrame with market data
        period: Lookback period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        ma_type: 'sma' (default) or 'ema' for middle band calculation
        
    Returns:
        Dict[str, pd.Series]: {
            'upper': Upper band series,
            'middle': MA series (SMA/EMA),
            'lower': Lower band series
        }
    """
    if len(df) < period:
        empty_series = pd.Series(np.nan, index=df.index)
        return {'upper': empty_series, 'middle': empty_series, 'lower': empty_series}
    
    if 'close' not in df.columns:
        empty_series = pd.Series(np.nan, index=df.index)
        return {'upper': empty_series, 'middle': empty_series, 'lower': empty_series}
    
    # Calculate middle band (SMA or EMA)
    if ma_type == 'ema':
        middle_band = df['close'].ewm(span=period, adjust=False).mean()
    else:  # default SMA
        middle_band = df['close'].rolling(window=period).mean()
    
    # Calculate standard deviation (always based on SMA)
    rolling_std = df['close'].rolling(window=period).std()
    
    return {
        'upper': middle_band + (num_std * rolling_std),
        'middle': middle_band,
        'lower': middle_band - (num_std * rolling_std)
    }


@indicator(name='vwap', precompute=False, vectorized=True)
def _vwap(df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
    """
    Internal helper function for Volume Weighted Average Price (VWAP)
    
    Args:
        df: Input DataFrame with 'close' and 'volume' columns
        period: If None (default), calculates cumulative VWAP.
                If specified, calculates rolling period VWAP.
    
    Returns:
        pd.Series: VWAP values (vectorized)
    """
    # Input validation
    if 'close' not in df.columns or 'volume' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    # Calculate price * volume
    pv = df['close'] * df['volume']
    
    if period:
        # Rolling VWAP
        if len(df) < period:
            return pd.Series(np.nan, index=df.index)
            
        pv_sum = pv.rolling(window=period).sum()
        vol_sum = df['volume'].rolling(window=period).sum()
    else:
        # Cumulative VWAP
        pv_sum = pv.cumsum()
        vol_sum = df['volume'].cumsum()
    
    # Calculate VWAP (with division by zero protection)
    vwap = pv_sum / vol_sum.replace(0, np.nan)  # Avoid division by zero
    
    return vwap

@indicator(name='obv', precompute=False, vectorized=True)
def _obv(df: pd.DataFrame) -> pd.Series:
    """
    Internal helper function for On-Balance Volume (OBV)
    
    Args:
        df: Input DataFrame with 'close' and 'volume' columns
    
    Returns:
        pd.Series: Cumulative OBV values
    """
    if len(df) < 2 or 'volume' not in df.columns or 'close' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    price_diff = df['close'].diff()
    volume_sign = np.sign(price_diff)
    volume_sign[price_diff == 0] = 0  # No change means 0 volume contribution
    
    obv = (df['volume'] * volume_sign).cumsum()
    return obv

@indicator(name='parabolic_sar', precompute=False, vectorized=True)
def _parabolic_sar(
    df: pd.DataFrame,
    step: float = 0.02,
    max_step: float = 0.2
) -> pd.Series:
    """
    Internal helper function for Parabolic SAR (Stop and Reverse)
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        step: Acceleration factor increment (default: 0.02)
        max_step: Maximum acceleration factor (default: 0.2)
        
    Returns:
        pd.Series: SAR values for each period
    """
    if len(df) < 2:
        return pd.Series(np.nan, index=df.index)
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    sar = np.full(len(df), np.nan)
    sar[0] = low[0]
    
    ep = high[0]
    af = step
    long = True
    
    for i in range(1, len(df)):
        prev_sar = sar[i-1]
        
        if long:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i-1], low[i])
            
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
                
            if low[i] < sar[i]:
                long = False
                sar[i] = ep
                ep = low[i]
                af = step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i-1], high[i])
            
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
                
            if high[i] > sar[i]:
                long = True
                sar[i] = ep
                ep = high[i]
                af = step
    
    return pd.Series(sar, index=df.index)
