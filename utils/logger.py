import logging
from logging.handlers import RotatingFileHandler
import os
from config import LOGGING_CONFIG

def setup_logger(name: str) -> logging.Logger:
    """
    Configure the application logger.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_CONFIG["LOG_LEVEL"])
    
    # Set log format
    formatter = logging.Formatter(LOGGING_CONFIG["LOG_FORMAT"])
    
    # Set file handler
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, LOGGING_CONFIG["LOG_FILE"]),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Set console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create application logger instance
app_logger = setup_logger("retail_analysis") 