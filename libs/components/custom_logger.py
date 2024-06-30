import logging
import os

def get_custom_logger(logger_name, logs_dir="", child_logger_name=""):
    if len(child_logger_name) == 0:
        logger = logging.getLogger(logger_name)

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_path = os.path.join(logs_dir, "{}.log".format(logger_name))

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger = logging.getLogger("{}.{}".format(logger_name, child_logger_name))

    return logger
