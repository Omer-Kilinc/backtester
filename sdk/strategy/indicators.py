import pandas as pd
import numpy as np
from typing import Dict
from sdk.strategy.base import indicator

# TODO: Write docstrings? Don't think it's necessary, because the functions are only called by the backtester, not the user. The user accesses them via the dataframe.


@indicator(name='sma', precompute=False)
def sma(df: pd.DataFrame, period: int = 20) -> float:
    if len(df) < period:
        return np.nan
    return df['close'].iloc[-period:].mean()


@indicator(name='ema', precompute=False)
def ema(df: pd.DataFrame, period: int = 20) -> float:
    if len(df) < period:
        return np.nan
    return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]

@indicator(name='rsi', precompute=False)
def rsi(df: pd.DataFrame, period: int = 14) -> float:
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

@indicator(name='macd', precompute=False)
def macd(df: pd.DataFrame,
            fast_period: int = 12,
            slow_period: int = 26,
            signal_period: int = 9) -> Dict[str, float]:
    if len(df) < slow_period + signal_period:
        return {'macd': np.nan, 'signal': np.nan, 'histogram': np.nan}

    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'macd': macd_line.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1]
    }

@indicator(name='atr', precompute=False)
def atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return np.nan
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]

@indicator(name='bollinger_bands', precompute=False)
def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> Dict[str, float]:
    if len(df) < period:
        return {'upper': np.nan, 'middle': np.nan, 'lower': np.nan}
    rolling_mean = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return {
        'upper': upper.iloc[-1],
        'middle': rolling_mean.iloc[-1],
        'lower': lower.iloc[-1]
    }

@indicator(name='vwap', precompute=False)
def vwap(df: pd.DataFrame, period: int = None) -> float:
    if 'volume' not in df.columns:
        return np.nan
    if period:
        pv = (df['close'] * df['volume']).rolling(period).sum()
        vol = df['volume'].rolling(period).sum()
    else:  
        pv = (df['close'] * df['volume']).cumsum()
        vol = df['volume'].cumsum()
    
    # Handle division by zero
    if vol.iloc[-1] == 0:
        return np.nan
    return (pv / vol).iloc[-1]

@indicator(name='obv', precompute=False)
def obv(df: pd.DataFrame) -> float:
    if len(df) < 2 or 'volume' not in df.columns:
        return np.nan
    obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    return obv.iloc[-1]

@indicator(name='parabolic_sar', precompute=False)
def parabolic_sar(df: pd.DataFrame,
                    step: float = 0.02,
                    max_step: float = 0.2) -> float:
    if len(df) < 2:
        return np.nan
    high = df['high']
    low = df['low']
    close = df['close']

    
    sar = low.iloc[0]
    ep = high.iloc[0]
    af = step
    long = True  # We assume that the initial position is a long (buy) position, which may not necessarily be true. However without this we cannot derive a value for the first few bars hence this assumption has been made.

    for i in range(1, len(df)):
        prev_sar = sar
        if long:
            sar = sar + af * (ep - sar)
            sar = min(sar, low.iloc[i - 1], low.iloc[i])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
            if low.iloc[i] < sar:
                long = False
                sar = ep
                ep = low.iloc[i]
                af = step
        else:
            sar = sar + af * (ep - sar)
            sar = max(sar, high.iloc[i - 1], high.iloc[i])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
            if high.iloc[i] > sar:
                long = True
                sar = ep
                ep = high.iloc[i]
                af = step

    return sar
