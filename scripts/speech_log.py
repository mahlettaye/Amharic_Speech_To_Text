import logging



def log_writer(message):
    logger=logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')


    file_handler=logging.FileHandler('speech.log')

    stream_handler=logging.StreamHandler()
    stream_handler.setFormatter(formatter)


    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.exception(message)
