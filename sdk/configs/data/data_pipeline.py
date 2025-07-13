from pydantic import BaseModel, Field
from typing import Optional, Literal, Union, Annotated

from ..imports.api_handler import APIImportConfig
from ..price_simulation.gbm_simulator import GBMSimulatorConfig
from ..processing.transformer import DataTransformerConfig
from ..processing.cleaning import CleaningConfig
from ..processing.noise_generator import NoiseGeneratorConfig
from ..splitting.data_splitter import DataSplitterConfig
from ..processing.data_conversion import DataConversionConfig

class APIInputConfig(BaseModel):
    source_type: Literal["api"]
    import_config: APIImportConfig
    transformer_config: DataTransformerConfig = Field(default_factory=DataTransformerConfig)
    cleaning_config: CleaningConfig = Field(default_factory=CleaningConfig)

class SimulatedInputConfig(BaseModel):
    source_type: Literal["simulated"]
    price_simulator_config: GBMSimulatorConfig

InputConfig = Annotated[Union[APIInputConfig, SimulatedInputConfig], Field(discriminator="source_type")]

class DataPipelineConfig(BaseModel):
    input_config: InputConfig
    noise_generator_config: Optional[NoiseGeneratorConfig] = None
    data_splitter_config: DataSplitterConfig = Field(default_factory=DataSplitterConfig)
    data_conversion_config: Optional[DataConversionConfig] = None


from typing import Literal

def make_pipeline_config(
    source_type: Literal["api", "simulated"],
    import_config: Optional[APIImportConfig] = None,
    transformer_config: Optional[DataTransformerConfig] = None,
    cleaning_config: Optional[CleaningConfig] = None,
    price_simulator_config: Optional[GBMSimulatorConfig] = None,
    noise_generator_config: Optional[NoiseGeneratorConfig] = None,
    data_splitter_config: Optional[DataSplitterConfig] = None,
    data_conversion_config: Optional[DataConversionConfig] = None,
) -> DataPipelineConfig:
    if source_type == "api":
        if import_config is None:
            raise ValueError("import_config is required when source_type is 'api'")
        input_cfg = APIInputConfig(
            source_type="api",
            import_config=import_config,
            transformer_config=transformer_config or DataTransformerConfig(),
            cleaning_config=cleaning_config or CleaningConfig(),
        )
    elif source_type == "simulated":
        if price_simulator_config is None:
            raise ValueError("price_simulator_config is required when source_type is 'simulated'")
        input_cfg = SimulatedInputConfig(
            source_type="simulated",
            price_simulator_config=price_simulator_config
        )
    else:
        raise ValueError("Invalid source_type. Must be 'api' or 'simulated'.")

    return DataPipelineConfig(
        input_config=input_cfg,
        noise_generator_config=noise_generator_config,
        data_splitter_config=data_splitter_config or DataSplitterConfig(),
        data_conversion_config=data_conversion_config,
    )
