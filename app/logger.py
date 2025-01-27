import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_name, log_dir="logs", log_file="app.log", max_bytes=10 * 1024 * 1024, backup_count=3):
    """
    Set up and return a logger with rotating file handler and console handler.
    :param log_name: Name of the logger.
    :param log_dir: Directory where log files will be saved.
    :param log_file: The main log file name.
    :param max_bytes: Maximum file size before rotating.
    :param backup_count: Number of backup files to keep.
    :return: Configured logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # File handler with rotation
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(logging.DEBUG)

    # Console handler for real-time logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger("hypothesis_testing", log_file="hypothesis_testing.log")
