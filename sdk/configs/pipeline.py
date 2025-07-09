from pydantic import BaseModel, Field, Discriminator
from typing import Optional, Union, Annotated

from .imports.api_handler import APIImportConfig
from .imports.csv_handler import CSVImportConfig
from .price_simulation.bootstrapped_returns import BootstrapReturnSimulatorConfig
from .price_simulation.gbm_simulator import GBMSimulatorConfig
from .price_simulation.ornstein_uhlenbeck_simulator import OrnsteinUhlenbeckSimulatorConfig
from .cleaning import CleaningConfig
from .noise_generator import NoiseGeneratorConfig
from .data_splitter import DataSplitterConfig
from .data_conversion import DataConversionConfig
from .transformer import DataTransformerConfig

ImportConfig = Annotated[
    Union[APIImportConfig, CSVImportConfig],
    Discriminator("import_type"),
]

PriceSimulatorConfig = Annotated[
    Union[BootstrapReturnSimulatorConfig, GBMSimulatorConfig, OrnsteinUhlenbeckSimulatorConfig],
    Discriminator("simulator_type"),
]

class PipelineConfig(BaseModel):
    import_config: Optional[ImportConfig] = None
    price_simulator_config: Optional[PriceSimulatorConfig] = None
    transformer_config: DataTransformerConfig = Field(default_factory=DataTransformerConfig)
    cleaning_config: CleaningConfig = Field(default_factory=CleaningConfig)
    noise_generator_config: Optional[NoiseGeneratorConfig] = None
    data_splitter_config: DataSplitterConfig = Field(default_factory=DataSplitterConfig)
    data_conversion_config: Optional[DataConversionConfig] = None
