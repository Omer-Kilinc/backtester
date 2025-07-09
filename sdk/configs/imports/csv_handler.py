from pydantic import BaseModel, Field, Literal
from ..base import BaseConfig

class CSVImportConfig(BaseConfig):
    import_type: Literal['csv'] = 'csv'
    filepath: str
