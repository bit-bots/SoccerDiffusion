#!/usr/bin/env python3

import typing

if typing.TYPE_CHECKING:
    from argparse import Namespace

import os
import sys
from pathlib import Path

from rich.console import Console

from soccer_diffusion import DEFAULT_RESAMPLE_RATE_HZ, IMAGE_MAX_RESAMPLE_RATE_HZ, __version__
from soccer_diffusion.dataset import logger
from soccer_diffusion.dataset.cli.args import CLIArgs, CLICommand, DBCommand, ImportType
from soccer_diffusion.dataset.converters.game_state_converter.b_human_game_state_converter import (
    BHumanGameStateConverter,
)
from soccer_diffusion.dataset.converters.game_state_converter.bit_bots_game_state_converter import (
    BitBotsGameStateConverter,
)
from soccer_diffusion.dataset.converters.image_converter import (
    BHumanImageConverter,
    BitbotsImageConverter,
    ImageConverter,
)
from soccer_diffusion.dataset.converters.synced_data_converter import SyncedDataConverter
from soccer_diffusion.dataset.db import Database
from soccer_diffusion.dataset.imports.model_importer import ImportStrategy
from soccer_diffusion.dataset.resampling.max_rate_resampler import MaxRateResampler
from soccer_diffusion.dataset.resampling.original_rate_resampler import OriginalRateResampler
from soccer_diffusion.dataset.resampling.previous_interpolation_resampler import PreviousInterpolationResampler

err_console = Console(stderr=True)


def main():
    debug_mode = os.getenv("LOGLEVEL") == "DEBUG"

    try:
        logger.debug("Parsing CLI args...")
        args: Namespace = CLIArgs().parse_args()
        logger.debug(f"Parsed CLI args: {args}")

        if args.version:
            logger.info(f"running soccer_diffusion CLI v{__version__}")
            sys.exit(0)

        should_create_schema = args.command == CLICommand.DB and args.db_command == DBCommand.CREATE_SCHEMA
        db = Database(args.db_path).create_session(create_schema=should_create_schema)

        match args.command:
            case CLICommand.DB:
                match args.db_command:
                    case DBCommand.RECORDING2MCAP:
                        from soccer_diffusion.dataset.recording2mcap import recording2mcap

                        recording2mcap(db.session, args.recording, args.output_dir)

                    case DBCommand.DUMMY_DATA:
                        from soccer_diffusion.dataset.dummy_data import insert_dummy_data

                        insert_dummy_data(db.session, args.num_recordings, args.num_samples_per_rec, args.image_step)

            case CLICommand.IMPORT:
                from soccer_diffusion.dataset.imports.model_importer import ImportMetadata, ModelImporter

                import_strategy: ImportStrategy
                import_path: Path = Path(args.file)
                upper_image_converter: ImageConverter
                location: str = args.location

                match args.type:
                    case ImportType.BIT_BOTS:
                        from soccer_diffusion.dataset.imports.strategies.bit_bots import BitBotsImportStrategy

                        logger.info(f"Trying to import file '{args.file}' to database...")

                        simulated = False
                        if "simulation" in str(args.file) or "simulated" in str(args.file):
                            simulated = True

                        metadata = ImportMetadata(
                            allow_public=True,
                            team_name="Bit-Bots",
                            robot_type="Wolfgang-OP",
                            location=location,
                            simulated=simulated,
                        )
                        upper_image_converter = BitbotsImageConverter(MaxRateResampler(IMAGE_MAX_RESAMPLE_RATE_HZ))
                        game_state_converter = BitBotsGameStateConverter(OriginalRateResampler())
                        synced_data_converter = SyncedDataConverter(
                            PreviousInterpolationResampler(DEFAULT_RESAMPLE_RATE_HZ)
                        )
                        import_strategy = BitBotsImportStrategy(
                            metadata, upper_image_converter, game_state_converter, synced_data_converter
                        )

                    case ImportType.B_HUMAN:
                        from soccer_diffusion.dataset.imports.strategies.b_human import BHumanImportStrategy

                        metadata = ImportMetadata(
                            allow_public=False,
                            team_name="B-Human",
                            robot_type="NAO6",
                            location=location,
                            simulated=False,
                        )
                        upper_image_converter = BHumanImageConverter(MaxRateResampler(IMAGE_MAX_RESAMPLE_RATE_HZ))
                        lower_image_converter = BHumanImageConverter(MaxRateResampler(IMAGE_MAX_RESAMPLE_RATE_HZ))
                        game_state_converter = BHumanGameStateConverter(OriginalRateResampler())
                        synced_data_converter = SyncedDataConverter(
                            PreviousInterpolationResampler(DEFAULT_RESAMPLE_RATE_HZ)
                        )

                        import_strategy = BHumanImportStrategy(
                            metadata,
                            upper_image_converter,
                            lower_image_converter,
                            game_state_converter,
                            synced_data_converter,
                            args.caching,
                            args.video,
                        )

                    case _:
                        raise ValueError(f"Unknown import type: {args.type}")

                logger.info(f"Importing file '{import_path}' to database...")
                importer = ModelImporter(db, import_strategy)
                importer.import_to_db(import_path)

        sys.exit(0)
    except Exception as e:
        logger.error(e)
        err_console.print_exception(show_locals=debug_mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
