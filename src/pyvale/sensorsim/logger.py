import logging
import logging.handlers
import sys

class Logger:

    def __init__(
        self,
        logger_name: str = __name__,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger_name = logger_name
        self.logger = logger

    def make_logger(self) -> None:
        logger = logging.getLogger(self.logger_name)
        file_handler = logging.FileHandler("plotter_log.log", mode="a")
        logger.addHandler(file_handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger = logger

    def put_error(self) -> None:
        if self.logger is not None:
            self.logger.error("This error came from a Logger object")
