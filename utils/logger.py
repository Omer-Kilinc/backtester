# Logs most/all process information to the users selected output, implemented to ensure the singleton pattern

import logging
import os
from datetime import datetime
from typing import Optional

# TODO Ensure correctness of code
# FIXME TODO Figure out what to do with critical logs, currently they are only logged to file and no action is taken 

_logger_configured = False
_log_file_path = None

def setup_logging(base_name: str = "app", level=logging.INFO, log_dir: str = "logs") -> str:
    """
    Set up the global logging configuration. Should be called once at application startup.
    Returns the log file path.
    """
    global _logger_configured, _log_file_path
    
    if _logger_configured:
        return _log_file_path
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _log_file_path = os.path.join(log_dir, f"{base_name}_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)-8s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(_log_file_path),
            # logging.StreamHandler()  
        ]
    )
    
    _logger_configured = True
    return _log_file_path

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. Automatically sets up logging if not already configured.
    
    Args:
        name: Logger name. If None, uses the calling module's name.
    """
    if not _logger_configured:
        setup_logging()
    
    if name is None:
        # Automatically determine the calling module's name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)
