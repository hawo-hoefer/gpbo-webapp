import os
import sys
import logging


def get_logger(name: str) -> logging.Logger:
    log_level = logging.getLevelName(os.getenv("LOG_LEVEL") or "INFO")

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    for handler in logger.handlers:
        logger.removeHandler(handler)
    sh = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s | %(uuid)s | %(msg)s"
    )
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler("app.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
