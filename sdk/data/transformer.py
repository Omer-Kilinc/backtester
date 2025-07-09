import pandas as pd 

# TODO Ensure correctness of code

from sdk.configs.transformer import DataTransformerConfig

class DataTransformer:
    def __init__(self, config: DataTransformerConfig):
        self.config = config

    def transform(self, df):
        if self.config.user_mapping:
            return self._apply_user_mapping(df)
        else:
            return self._auto_detect_and_transform(df)

    def _apply_user_mapping(self, df):
        required_fields = ["timestamp", "open", "close"]
        optional_fields = ["high", "low", "volume"]

        standardized = {}
        for std_col in required_fields + optional_fields:
            user_col = self.config.user_mapping.get(std_col)
            if user_col in df.columns:
                standardized[std_col] = df[user_col]
            elif std_col in required_fields:
                raise ValueError(f"Missing required field: {std_col}")
            else:
                standardized[std_col] = None  

        # Convert to ISO 8601
        standardized["timestamp"] = pd.to_datetime(standardized["timestamp"], errors='coerce', utc=True)
        standardized["timestamp"] = standardized["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Include any extra columns the user might have
        for col in df.columns:
            if col not in self.config.user_mapping.values():
                standardized[col] = df[col]

        return pd.DataFrame(standardized)

    def _auto_detect_and_transform(self, df):
        column_keywords = {
            "timestamp": ["time", "date", "ts", "timestamp"],
            "open": ["open"],
            "high": ["high"],
            "low": ["low"],
            "close": ["close"],
            "volume": ["vol", "volume"]
        }

        mapping = {}
        for std_col, keywords in column_keywords.items():
            for col in df.columns:
                col_lower = col.lower()
                if any(kw.lower() in col_lower for kw in keywords):
                    mapping[std_col] = col
                    break  

        self.config.user_mapping = mapping
        return self._apply_user_mapping(df)