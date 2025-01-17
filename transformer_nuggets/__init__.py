from transformer_nuggets import quant as quant, utils as utils


def init_logging():
    """
    Configure logging for transformer_nuggets library at INFO level.
    Adds a StreamHandler if none exists.
    """
    import logging

    logger = logging.getLogger("transformer_nuggets")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
