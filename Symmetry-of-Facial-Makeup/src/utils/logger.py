import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()


def creat_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with the specified name, log file, and logging level.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_dir = os.getenv("LOG_DIR")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{timestamp}_{name}.log"))
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s :%(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    console_formatter = logging.Formatter(
        "[%(name)s] - [%(levelname)s] : %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
