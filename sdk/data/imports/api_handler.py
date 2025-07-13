import pandas as pd
import requests
from typing import Any, Dict, Optional, Literal
from pydantic import ValidationError, parse_obj_as

from sdk.configs.data.imports.api_handler import (APIImportConfig, BinanceAPIImportConfig, 
                                              CustomAPIImportConfig, BinanceParams)
from utils.logger import get_logger

logger = get_logger(__name__)


class APIImportHandler:
    def __init__(self, config: APIImportConfig):
        self.config = config

    def fetch_data(self) -> pd.DataFrame:
        logger.info(f"Fetching data using api_type: {self.config.api_type}")
        if self.config.api_type == 'binance':
            return self._fetch_binance()
        elif self.config.api_type == 'custom':
            return self._fetch_custom()
        else:
            # This path should ideally not be reached due to Pydantic's validation
            raise ValueError(f"Unsupported api_type: {self.config.api_type}")

    def _fetch_binance(self) -> pd.DataFrame:
        logger.info(f"Fetching data from Binance API for symbol {self.config.binance_params.symbol}")
        params = self.config.binance_params.model_dump(exclude_none=True)
        
        try:
            response = requests.get(str(self.config.binance_url), params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API request failed: {e}")
            raise

        data = response.json()
        
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        df = pd.DataFrame(data, columns=columns)
        df.drop(columns=["ignore"], inplace=True)
        
        for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", 
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        
        logger.info(f"Fetched and parsed {len(df)} rows from Binance API")
        return df

    def _fetch_custom(self) -> pd.DataFrame:
        logger.info(f"Fetching data from Custom API: {self.config.url}")
        data_path = self.config.data_path.split('.')

        try:
            if self.config.method == "GET":
                response = requests.get(str(self.config.url), headers=self.config.headers, params=self.config.params)
            else: # POST
                response = requests.post(str(self.config.url), headers=self.config.headers, json=self.config.params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Custom API request failed: {e}")
            raise

        raw_data = response.json()
        try:
            for key in data_path:
                raw_data = raw_data[key]
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to access data path '{self.config.data_path}' in API response: {e}")
            raise

        try:
            parsed = [
                {
                    key: item.get(self.config.mapping[key]) 
                    for key in self.config.mapping
                }
                for item in raw_data
            ]
            df = pd.DataFrame(parsed)
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to map API response to DataFrame: {e}")
            raise

        logger.info(f"Fetched and parsed {len(df)} rows from Custom API")
        return df


def api_import_data(
    *,
    source: Literal['binance', 'custom'],
    # Binance specific parameters
    symbol: Optional[str] = None,
    interval: Literal['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'] = '1m',
    startTime: Optional[int] = None,
    endTime: Optional[int] = None,
    timeZone: Optional[str] = '0',
    limit: Optional[int] = 500,
    # Custom API specific parameters
    url: Optional[str] = None,
    method: Literal['GET', 'POST'] = 'GET',
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    mapping: Optional[Dict[str, str]] = None,
    data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    User-facing function to fetch market data from an API (Binance or custom).

    This function validates inputs, selects the appropriate API handler, fetches the
    data, and returns it as a pandas DataFrame.

    Args:
        source: The API source to use, either 'binance' or 'custom'.
        symbol: (Binance) The trading pair symbol (e.g., 'BTCUSDT'). Required for Binance.
        interval: (Binance) The kline interval.
        startTime: (Binance) The start timestamp in milliseconds.
        endTime: (Binance) The end timestamp in milliseconds.
        timeZone: (Binance) The timezone offset.
        limit: (Binance) The number of data points to retrieve (1-1000).
        url: (Custom) The API endpoint URL. Required for custom source.
        method: (Custom) The HTTP method to use ('GET' or 'POST').
        headers: (Custom) The request headers.
        params: (Custom) The request parameters or body.
        mapping: (Custom) A mapping from internal column names to API field names. 
                 Required for custom source.
        data_path: (Custom) A dot-separated path to the data in the JSON response. 
                   Required for custom source.

    Returns:
        A pandas DataFrame containing the fetched market data.

    Raises:
        ValueError: If required parameters for the selected source are missing.
        pydantic.ValidationError: If any of the parameters fail validation.
        requests.exceptions.RequestException: If the API request fails.
    """
    config_data = locals()
    config_data['api_type'] = source

    try:
        if source == 'binance':
            if not symbol:
                raise ValueError("Parameter 'symbol' is required for Binance source.")
            binance_params = {k: v for k, v in config_data.items() if k in BinanceParams.model_fields}
            config = BinanceAPIImportConfig(binance_params=binance_params)
        
        elif source == 'custom':
            required_params = ['url', 'mapping', 'data_path']
            if any(config_data.get(p) is None for p in required_params):
                raise ValueError(f"Parameters {required_params} are required for custom source.")
            config = CustomAPIImportConfig(**config_data)

        else:
            raise ValueError(f"Unsupported source: '{source}'. Must be 'binance' or 'custom'.")

        handler = APIImportHandler(config)
        return handler.fetch_data()

    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise
    except (ValueError, requests.exceptions.RequestException) as e:
        logger.error(f"An error occurred during data import: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise
