import logging


def setup_logging(level: str = "INFO") -> None:
    # One line per event, key=value style, so logs are easy to grep by job id.
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
    )
