import logging
import logging.handlers
import sys

class Logger():

    def __init__(self,
                 logger_name: str = __name__,
                 logger = None):
        
        self.logger_name = logger_name

    def make_logger(self):
        logger = logging.getLogger(self.logger_name)
        file_handler = logging.FileHandler("plotter_log.log", mode = 'a')
        logger.addHandler(file_handler)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger = logger

    def put_error(self):
        self.logger.error("This error came from a Logger object")
