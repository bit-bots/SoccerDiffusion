from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from soccer_diffusion.dataset import logger
from soccer_diffusion.dataset.models import Base


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA temp_store=MEMORY")
    logger.info("Set SQLite to run in write-ahead logging mode, improving parallel write performance")
    cursor.close()


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.engine: Engine = self._setup_sqlite()
        self.session: Session

    def __del__(self):
        self.close_session()
        self.engine.dispose()
        logger.info("Database connection closed")

    def _setup_sqlite(self) -> Engine:
        return create_engine(f"sqlite:///{self.db_path}")

    def _create_schema(self) -> None:
        logger.info("Creating database schema")
        Base.metadata.create_all(self.engine)
        logger.info("Database schema created")

    def create_session(self, create_schema: bool = True) -> "Database":
        logger.info("Setting up database session")
        if create_schema:
            self._create_schema()

        self.session = sessionmaker(bind=self.engine)()
        logger.info("Database session created")

        return self

    def close_session(self) -> "Database":
        if self.session:
            self.session.close_all()
            logger.info("Database session closed")
        else:
            logger.warning("No database session to close")

        return self

    def clear_database(self) -> "Database":
        logger.info("Clearing database")
        Base.metadata.drop_all(self.engine)
        logger.info("Database cleared")

        return self
