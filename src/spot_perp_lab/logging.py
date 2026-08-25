"""Application logging setup."""

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure concise console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
