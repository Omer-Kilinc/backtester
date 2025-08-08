"""
Optimization orchestrator - main entry point for all optimization workflows.

The orchestrator manages the optimization lifecycle and coordinates between:
- Multiple backtester instances with different parameter configurations
- Data pipeline for train/test/validation splits  
- Optimization algorithms (grid search, random search, bayesian)
- Result aggregation and overfitting detection
- Progress tracking and cancellation capabilities
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
import pandas as pd
from pathlib import Path
import json

from sdk.backtester.backtester import Backtester
from sdk.configs.backtester.backtester import BacktesterConfig
from sdk.data.data_pipeline import DataPipeline, DataPipelineConfig
from sdk.configs.optimization.orchestrator import OptimizationConfig
from sdk.strategy.base import Strategy
from sdk.configs.analytics.analytics import AnalyticsConfig
from .algorithms.base import BaseOptimizer, OptimizationResult
from .search_space.space import SearchSpace
from .results.aggregator import ResultAggregator
from .overfitting_detection import OverfittingDetector
from utils.logger import get_logger

logger = get_logger(__name__)

class OptimizationOrchestrator:
    """
    Main orchestrator for optimization workflows.
    
    Manages the complete optimization lifecycle:
    1. Initialization - Parameter space definition and algorithm selection
    2. Execution - Parallel backtester runs with different parameter configurations
    3. Finalization - Result aggregation, analysis, and overfitting detection
    
    Features:
    - Error recovery and partial failure scenarios
    - Progress tracking and cancellation capabilities
    - Integration with existing data pipeline and analytics
    - Multiple optimization algorithm support
    """

    def __init__(
        self,
        strategy_class: type,
        data_pipeline_config: DataPipelineConfig,
        base_backtester_config: BacktesterConfig,
        analytics_config: Optional[AnalyticsConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize the optimization orchestrator.
        
        Args:
            strategy_class: Strategy class to optimize (not instance)
            data_pipeline_config: Configuration for data preparation
            base_backtester_config: Base configuration for backtester
            analytics_config: Configuration for analytics
            optimization_config: Configuration for optimization process
        """

        self.strategy_class = strategy_class
        self.data_pipeline_config = data_pipeline_config
        self.base_backtester_config = base_backtester_config
        self.analytics_config = analytics_config
        self.config = optimization_config or OptimizationConfig()
        
        # State management
        self.is_running = False
        self.is_cancelled = False
        self._cancel_event = threading.Event()
        self._progress_callbacks: List[Callable] = []
        
        # Results storage
        self.optimization_results: List[OptimizationResult] = []
        self.failed_runs: List[Dict[str, Any]] = []
        
        # Components
        self.data_pipeline = DataPipeline(self.data_pipeline_config)
        self.result_aggregator = ResultAggregator()
        self.overfitting_detector = OverfittingDetector(
            threshold=self.config.overfitting_threshold
        )

        self._prepared_data: Optional[List[Dict[str, pd.DataFrame]]] = None

    # These are used as the task will be asynchronous
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)

    def _notify_progress(self, progress_data: Dict[str, Any]):
        """Notify all progress callbacks."""
        if not self.config.enable_progress_tracking:
            return
            
        for callback in self._progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def prepare_data(self) -> List[Dict[str, pd.DataFrame]]:
        """
        Prepare data using the existing data pipeline.
        
        Returns:
            List of dicts containing train/test/validation splits
        """
        if self._prepared_data is not None:
            logger.info("Using cached prepared data")
            return self._prepared_data
            
        logger.info("Preparing data for optimization...")
        
        # Run the data pipeline to get processed data
        pipeline_result = self.data_pipeline.run_standard_pipeline()
        
        data_splits = []
        
        if isinstance(pipeline_result, pd.DataFrame):
            # Single DataFrame - treat as training data only
            logger.info("Processing single DataFrame as training data")
            data_splits = [{
                'train': pipeline_result,
                'test': None,
                'validation': None
            }]
            
        elif isinstance(pipeline_result, tuple) and len(pipeline_result) == 2:
            # Train/test pair from standard splitting
            train_data, test_data = pipeline_result
            logger.info("Processing train/test pair from standard splitting")
            data_splits = [{
                'train': train_data,
                'test': test_data,
                'validation': None  # No validation split available
            }]
            
        elif isinstance(pipeline_result, list) and all(isinstance(item, tuple) and len(item) == 2 for item in pipeline_result):
            # Walk-forward splits - multiple train/test pairs
            if all(isinstance(pair[0], pd.DataFrame) for pair in pipeline_result):
                # List[Tuple[DataFrame, DataFrame]] - raw DataFrames
                logger.info(f"Processing {len(pipeline_result)} walk-forward DataFrame splits")
                data_splits = [
                    {
                        'train': train_df,
                        'test': test_df,
                        'validation': None  # No validation available in walk-forward
                    }
                    for train_df, test_df in pipeline_result
                ]
            elif all(isinstance(pair[0], str) for pair in pipeline_result):
                # List[Tuple[str, str]] - file paths
                logger.info(f"Processing {len(pipeline_result)} walk-forward file path splits")
                data_splits = []
                for train_path, test_path in pipeline_result:
                    train_df = pd.read_parquet(train_path)
                    test_df = pd.read_parquet(test_path)
                    data_splits.append({
                        'train': train_df,
                        'test': test_df,
                        'validation': None
                    })
            else:
                raise ValueError("Mixed types in walk-forward results not supported")
                
        elif isinstance(pipeline_result, str):
            # Single file path - load and treat as training data
            logger.info("Processing single file path as training data")
            full_data = pd.read_parquet(pipeline_result)
            data_splits = [{
                'train': full_data,
                'test': None,
                'validation': None
            }]
            
        else:
            raise ValueError(f"Unexpected pipeline result type: {type(pipeline_result)}")
            
        # Cache the prepared data
        self._prepared_data = data_splits
        
        # Log summary of prepared data
        total_splits = len(data_splits)
        logger.info(f"Data prepared: {total_splits} split(s)")
        for i, split in enumerate(data_splits):
            train_len = len(split['train']) if split['train'] is not None else 0
            test_len = len(split['test']) if split['test'] is not None else 0
            val_len = len(split['validation']) if split['validation'] is not None else 0
            logger.info(f"  Split {i}: Train={train_len}, Test={test_len}, Validation={val_len}")
        
        return data_splits

    def optimize(
        self,
        search_space: SearchSpace,
        optimizer: BaseOptimizer,
        objective_metric: str = 'sharpe_ratio',
        n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete optimization workflow.
        
        Args:
            search_space: Parameter search space definition
            optimizer: Optimization algorithm to use
            objective_metric: Metric to optimize (from analytics)
            n_trials: Number of trials (if not specified in optimizer)
            
        Returns:
            Dict containing optimization results and analysis
        """
        pass

    
    
        
        