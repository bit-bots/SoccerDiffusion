import logging
import os

from rich.logging import RichHandler

from ddlitlab2024 import LOGGING_PATH, SESSION_ID

MODULE_NAME: str = "dataset"

# Init logging
logging.basicConfig(
    filename=LOGGING_PATH,
    encoding="utf-8",
    level=logging.DEBUG,
    format=f"%(asctime)s | {SESSION_ID} | %(name)s:%(levelname)s: %(message)s",
)

# Create additional logging config for the shell with configurable log level
console = RichHandler(
    log_time_format="%H:%M:%S",
    show_path=False,
    rich_tracebacks=True,
    tracebacks_show_locals=True,
)
console.setLevel(os.environ.get("LOGLEVEL", "INFO"))

logger = logging.getLogger(MODULE_NAME)
logger.addHandler(console)
