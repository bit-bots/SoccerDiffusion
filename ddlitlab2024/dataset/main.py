#!/usr/bin/env python3

import os
import sys

from rich.console import Console

from ddlitlab2024 import __version__
from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.cli import CLIArgs, CLICommand
from ddlitlab2024.dataset.db import Database

err_console = Console(stderr=True)


def main():
    debug_mode = os.getenv("LOGLEVEL") == "DEBUG"

    try:
        logger.debug("Parsing CLI args...")
        args: CLIArgs = CLIArgs().parse_args()
        if args.version:
            logger.info(f"running ddlitlab2024 CLI v{__version__}")
            sys.exit(0)

        if args.command == CLICommand.DB:
            db = Database(args.db_path).create_session(args.create_schema)
            logger.info(f"Database session created: {db}")

        logger.info(f"CLI args: {args}")
        sys.exit(0)
    except Exception as e:
        logger.error(e)
        err_console.print_exception(show_locals=debug_mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
