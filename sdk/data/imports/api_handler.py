from utils.logger import setup_logging, get_logger  
import pandas as pd
import requests
from sdk.configs.imports.api_handler import APIImportConfig, BinanceParams, CustomAPIImportConfig
from pydantic import TypeAdapter

# TODO Ensure correctness of code
# TODO Remove redundant imports

class APIImportHandler:
    """
    Base class for API import handlers
    """
    def __init__(self, config: APIImportConfig):
        """
        Initialize the API import handler
        """
        self.config = config
    
    def fetch_data(self):
        """
        Fetch data from the API
        """
        raise NotImplementedError
    
    def _fetch_binance(self):
        logger = get_logger(__name__)
        logger.info("Fetching data from Binance API")

        logger.debug(f"Params: {self.config.binance_params}")
        params = self.config.binance_params.model_dump(exclude_none=True)
        res = requests.get(self.config.binance_url, params=params)
        if res.status_code != 200:
            logger.error(f"Binance API error: {res.status_code} {res.text}")
            res.raise_for_status()

        data = res.json()
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        df = pd.DataFrame(data, columns=columns)
        df.drop(columns=["ignore"], inplace=True)

        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        logger.info(f"Fetched {len(df)} rows from Binance API")
        return df
    
    def _fetch_custom(self):
        url = self.config.url
        method = self.config.method
        headers = self.config.headers
        params = self.config.params
        mapping = self.config.mapping
        data_path = self.config.data_path.split(".")

        response = requests.get(url, headers=headers, params=params) if method == "GET" else requests.post(url, headers=headers, data=params)
        raw_data = response.json()

        for key in data_path:
            raw_data = raw_data[key]

        parsed = [
            {
                "timestamp": item[mapping["timestamp"]],
                "open": item[mapping["open"]],
                "high": item[mapping["high"]],
                "low": item[mapping["low"]],
                "close": item[mapping["close"]],
                "volume": item[mapping["volume"]]
            }
            for item in raw_data
        ]
        return pd.DataFrame(parsed)


# Ignore, testing for future use. 
# raw = {
#     "import_type": "api",
#     "api_type": "binance",
#     "binance_url": "https://api.binance.com/api/v3/klines",
#     "binance_params": {
#         "symbol": "BTCUSDT",
#         "interval": "1m",
#         "limit": 500
#     }
# }

# config = TypeAdapter(APIImportConfig).validate_python(raw)
# test = APIImportHandler(config)
# print(test._fetch_binance())


