import logging
import os

from ddlitlab2024 import LOGGING_PATH, SESSION_ID

# Init logging
logging.basicConfig(
    filename=LOGGING_PATH,
    encoding="utf-8",
    level=logging.DEBUG,
    format=f"%(asctime)s | {SESSION_ID} | %(name)s:%(levelname)s: %(message)s",
)

# Create additional logging config for the shell with configurable log level
console = logging.StreamHandler()
console.setLevel(os.environ.get("LOGLEVEL", "INFO"))
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger = logging.getLogger("dataset")
logger.addHandler(console)
