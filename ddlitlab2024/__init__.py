import importlib.metadata
import os
import sys
from uuid import UUID, uuid4

_project_name: str = "ddlitlab2024"
__version__: str = importlib.metadata.version(_project_name)

# Craft LOGGING PATH
# Get the log directory from the environment variable or use the default
_logging_dir: str = os.environ.get("DDLITLAB_LOG_DIR", "../logs")

# Verify that the log directory exists and create it if it doesn't
if not os.path.exists(_logging_dir):
    try:
        os.makedirs(_logging_dir)
    except OSError:
        print(f"ERROR: Failed to create log directory {_logging_dir}. Exiting.")
        sys.exit(1)

_logging_path: str = os.path.join(_logging_dir, f"{_project_name}.log")

# Create log file if it doesn't exist or verify that it is writable
try:
    with open(_logging_path, "a"):
        pass
except OSError:
    print(f"ERROR: Failed to create or open log file {_logging_path}. Exiting.")
    sys.exit(1)

LOGGING_PATH: str = _logging_path

SESSION_ID: UUID = uuid4()
