import logging


NAME_LOGGER = "datacube_logger"
logger = logging.getLogger(NAME_LOGGER)

DEFAULT_LOGGER_LEVEL = 20
logger.setLevel(DEFAULT_LOGGER_LEVEL)

# create file handler which logs even debug messages
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)-15s - %(file_name)s - [%(levelname)s] - %(message)s",
    "%m/%d/%Y %I:%M:%S %p",
)
ch.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.propagate = False


class Logger(logging.Logger):
    def __init__(self, name):
        self.d = {"file_name": name}

    def debug(self, message):
        logger.debug(message, extra=self.d)

    def info(self, message):
        logger.info(message, extra=self.d)

    def error(self, message):
        logger.error(message, extra=self.d)

    def warning(self, message):
        logger.warning(message, extra=self.d)

    def critical(self, message):
        logger.critical(message, extra=self.d)

    def non_verbose(self):
        logger.setLevel(logging.INFO)
