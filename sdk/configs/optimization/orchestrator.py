from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    max_concurrent_runs: int = 4
    timeout_per_run: Optional[float] = None
    save_intermediate_results: bool = True
    results_dir: Optional[Path] = None
    enable_progress_tracking: bool = True
    overfitting_threshold: float = 0.15  # 15% performance degradation threshold between test and validation # TODO: Should this stay?