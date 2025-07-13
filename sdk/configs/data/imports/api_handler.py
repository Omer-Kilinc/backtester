import re
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Dict, Any, Literal, Optional, Union, Annotated
from ..base import BaseConfig


class BinanceParams(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g. BTCUSDT)")
    interval: Literal[
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    ] = Field('1m', description="Kline interval for data")
    startTime: Optional[int] = Field(None, ge=0, description="Unix timestamp in milliseconds (UTC) for start of data")
    endTime: Optional[int] = Field(None, ge=0, description="Unix timestamp in milliseconds (UTC) for end of data")
    timeZone: Optional[str] = Field('0', description="Timezone offset (e.g. '-1:00', '5:45', '0') Default is UTC")
    limit: Optional[int] = Field(500, ge=1, le=1000, description="Max number of results to return. Max 1000.")

    @validator('symbol')
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper()

    @validator('timeZone')
    def validate_timezone(cls, v: str) -> str:
        if v is not None:
            if not re.match(r'^[+-]?(\d{1,2}(:\d{2})?|0)$', v):
                raise ValueError('Invalid timeZone format. Use offset like "-1:00", "5:45", or "0" for UTC.')
        return v


class BinanceAPIImportConfig(BaseModel):
    import_type: Literal['api'] = 'api'
    api_type: Literal['binance'] = 'binance'
    binance_url: HttpUrl = Field("https://api.binance.com/api/v3/klines", description="Binance API endpoint URL")
    binance_params: BinanceParams = Field(..., description="Binance API query parameters grouped together")


class CustomAPIImportConfig(BaseModel):
    import_type: Literal['api'] = 'api'
    api_type: Literal['custom'] = 'custom'
    url: HttpUrl = Field(..., description="API endpoint URL")
    method: Literal['GET', 'POST'] = 'GET'
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    mapping: Dict[str, str] = Field(..., description="Mapping from API response fields to internal fields")
    data_path: str = Field(..., description="JSON path to data within API response")

    @validator('mapping')
    def validate_mapping(cls, v: Dict[str, str]) -> Dict[str, str]:
        required_keys = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f"Mapping must contain the following keys: {required_keys}")
        return v


APIImportConfig = Annotated[
    Union[BinanceAPIImportConfig, CustomAPIImportConfig],
    Field(discriminator='api_type')
]
