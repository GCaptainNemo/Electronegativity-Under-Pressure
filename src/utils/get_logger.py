import logging


def get_logger(filename="importance_output.log", name="importance_logger"):
    """
    Input Args:
        filename: output file
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    sh1 = logging.StreamHandler()
    sh1.setLevel(logging.WARNING)
    fh1 = logging.FileHandler(filename=filename, mode='w')
    fmt1 = logging.Formatter(fmt=
                             "%(asctime)s - %(levelname)-9s - "
                             "%(filename)-8s : %(lineno)s line - "
                             "%(message)s")
    sh1.setFormatter(fmt1)
    fh1.setFormatter(fmt1)
    logger.addHandler(sh1)
    logger.addHandler(fh1)
    return logger