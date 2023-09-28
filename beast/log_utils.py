import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    cyan = "\u001b[36m"
    green = "\u001b[32m"
    yellow = "\u001b[33m"
    red = "\u001b[35m"
    bold_red = "\u001b[31m"
    reset = "\u001b[0m"
    debug_format = "%(asctime)s - %(name)s - {colour}%(levelname)s - %(message)s\u001b[0m (%(filename)s:%(lineno)d)"
    norm_format = "%(asctime)s - %(name)s - {colour}%(levelname)s - %(message)s\u001b[0m"

    FORMATS = {
        logging.DEBUG: debug_format.format(colour=cyan),
        logging.INFO: norm_format.format(colour=green),
        logging.WARNING: debug_format.format(colour=yellow),
        logging.ERROR: debug_format.format(colour=red),
        logging.CRITICAL: debug_format.format(colour=bold_red)
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def get_logger(module_name, output_file=None, terminal_level=logging.DEBUG, file_level=logging.DEBUG):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(terminal_level)
    terminal_handler.setFormatter(CustomFormatter())
    logger.addHandler(terminal_handler)

    if not output_file is None:
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(file_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
