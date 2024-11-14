#!/usr/bin/env python3

import typing

if typing.TYPE_CHECKING:
    from argparse import Namespace

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
        args: Namespace = CLIArgs().parse_args()
        logger.debug(f"Parsed CLI args: {args}")

        if args.version:
            logger.info(f"running ddlitlab2024 CLI v{__version__}")
            sys.exit(0)

        if args.command == CLICommand.DB:
            create_schema = args.db_command == "create-schema" or args.db_command == "dummy-data"
            db: Database = Database(args.db_path).create_session(create_schema=create_schema)

            match args.db_command:
                case "recording2mcap":
                    from ddlitlab2024.dataset.recording2mcap import recording2mcap
                    recording2mcap(db.session, args.recording, args.output_dir)

                case "dummy-data":
                    from ddlitlab2024.dataset.dummy_data import insert_dummy_data
                    insert_dummy_data(db.session, args.num_recordings, args.num_samples_per_rec, args.image_step)

                    insert_dummy_data(db, args.num_recordings, args.num_samples_per_rec, args.image_step)

        sys.exit(0)
    except Exception as e:
        logger.error(e)
        err_console.print_exception(show_locals=debug_mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
