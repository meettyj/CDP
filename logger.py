import logging
import datetime


class CustomFormatter(logging.Formatter):
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Create custom logger logging all five levels
logging.getLogger().handlers.clear()
logger_global = logging.getLogger("")
logger_global.setLevel(logging.DEBUG)

# Define format for logs
fmt = '%(asctime)s - %(levelname)s - %(message)s'

# Create stdout handler for logging to the console (logs all five levels)
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(CustomFormatter(fmt))

# Create file handler for logging to a file (logs all five levels)
# today = datetime.date.today()
# file_handler = logging.FileHandler('my_app_{}.log'.format(today.strftime('%Y_%m_%d')))
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(logging.Formatter(fmt))

# Add both handlers to the logger
logger_global.addHandler(stdout_handler)
# logger_global.addHandler(file_handler)
