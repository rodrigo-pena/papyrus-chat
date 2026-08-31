"""Human-readable logging configuration shared by the command-line tools."""

import logging
import sys
from collections.abc import Callable

_HANDLER_NAME = "papyrus-chat-cli"


def configure_cli_logging(*, verbose: bool = False) -> Callable[[], None]:
    """Send package logs to stderr and return a function that restores prior state."""
    package_logger = logging.getLogger("papyrus_chat")
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    package_logger.propagate = False

    for handler in list(package_logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            package_logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    package_logger.addHandler(handler)

    def restore() -> None:
        package_logger.removeHandler(handler)
        handler.close()
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate

    return restore
