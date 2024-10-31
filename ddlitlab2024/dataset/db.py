from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.models import Base


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine: Engine = self._setup_sqlite()
        self.session: Session | None = None

    def _setup_sqlite(self) -> Engine:
        return create_engine(f"sqlite:///{self.db_path}")

    def _create_schema(self) -> None:
        logger.info("Creating database schema")
        Base.metadata.create_all(self.engine)
        logger.info("Database schema created")

    def create_session(self, create_schema: bool = True) -> Session:
        logger.info("Setting up database session")
        if create_schema:
            self._create_schema()
        return sessionmaker(bind=self.engine)()

    def close_session(self) -> None:
        if self.session:
            self.session.close()
            logger.info("Database session closed")
        else:
            logger.warning("No database session to close")
