import logging
import sys


def get_logger(log_file_path="training.log"):
    """
    Return a **root logger** object to record information in a log file.
    Use logging.info() anywhere to record information, since this is equivalent to logger.info() when logger is the root logger.

    Args:
        log_file_path (str, optional): The path of the log file, where logs will be saved.

    Returns:
        logger (logging.Logger): A logger object to record information in a log file.

    Example Usage:
    ```
    logger = get_logger('log.log') # Get the root logger
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.error("This is an error message")
    logging.warning("This is a warning message")
    logging.critical("This is a critical message")
    ```
    """
    logger = logging.getLogger() # Get the root logger
    logger.setLevel(logging.INFO)

    # Create and set a FileHandler (output to file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create and set a StreamHandler (output to console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create Formatter
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s", # log format
        datefmt="%Y-%m-%d %H:%M:%S" # time format
    )

    # bind the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    #sys.stdout = StreamToLogger(logger, logging.INFO)
    #sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger


def close_logger(logger):
    """Close loggers and free up resources (If not, handlers in different runs can overlap.)"""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)




# TODO: tqdm progress bar will also be recorded. Waiting for a fix.
class StreamToLogger:
    """a class to redirect stdout and stderr to a logger"""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        if message.strip() and not message == '[]':
            # BUG: Jupyter notebook seems to have a bug: when editing a cell, the logger will output "[]" to the console, so we add a check to disable it.
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass
