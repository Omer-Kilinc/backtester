from typing import Callable

INDICATOR_REGISTRY: dict[str, dict[str, Callable | bool]] = {}

def indicator(name=None, precompute=False):
    """Decorator factory to register an indicator function in the global registry.

    Registers a function as a financial indicator in `INDICATOR_REGISTRY`, allowing it to be
    used by strategies. The indicator can be optionally precomputed for performance.

    Args:
        name (str, optional): Custom name for the indicator. If `None`, the function's
            `__name__` is used. Defaults to `None`.
        precompute (bool, optional): If `True`, the indicator will be marked for precomputation
            in strategies that support it. Defaults to `False`.

    Returns:
        function: A decorator that registers the input function in `INDICATOR_REGISTRY`.

    Example:
        >>> @indicator(name="ema", precompute=True)
        >>> def exponential_moving_average(data):
        ...     # Compute EMA
        ...     return ema_values
        >>> # Now registered in INDICATOR_REGISTRY as:
        >>> # {"ema": {"func": exponential_moving_average, "precompute": True}}
    """
    def decorator(func):
        indicator_name = name or func.__name__
        INDICATOR_REGISTRY[indicator_name] = {
            'func': func,
            'precompute': precompute
        }
        return func
    return decorator