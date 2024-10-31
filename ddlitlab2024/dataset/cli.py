import sys
from enum import Enum
from pathlib import Path

from tap import Tap

from ddlitlab2024 import DB_PATH


class ImportType(str, Enum):
    ROS_BAG = "rosbag"


class CLICommand(str, Enum):
    DB = "db"
    IMPORT = "import"


class DBArgs(Tap):
    create_schema: bool = False

    def configure(self) -> None:
        self.add_argument(
            "create_schema",
            type=bool,
            help="Create the base database schema, if it doesn't exist",
            nargs="?",
        )


class ImportArgs(Tap):
    import_type: ImportType
    file: Path

    def configure(self) -> None:
        self.add_argument(
            "import-type",
            type=ImportType,
            help="Type of import to perform",
        )
        self.add_argument(
            "file",
            type=Path,
            help="File to import",
        )


class CLIArgs(Tap):
    dry_run: bool = False
    db_path: str = DB_PATH  # Path to the sqlite database file
    version: bool = False  # if set print version and exit

    def __init__(self):
        super().__init__(
            description="ddlitlab dataset CLI",
            underscores_to_dashes=True,
        )

    def configure(self) -> None:
        self.add_subparsers(dest="command", help="Command to run")
        self.add_subparser(CLICommand.DB.value, DBArgs, help="Database management commands")
        self.add_subparser(CLICommand.IMPORT.value, ImportArgs, help="Import data into the database")

    def print_help_and_exit(self) -> None:
        self.print_help()
        sys.exit(0)

    def process_args(self) -> None:
        if self.command == CLICommand.DB:
            all_args = (self.create_schema,)

            if not any(all_args):
                self.print_help_and_exit()
