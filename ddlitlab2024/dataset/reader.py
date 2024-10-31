import sqlite3

from ddlitlab2024 import DB_PATH


class Reader:
    def __init__(self, db_path: str = DB_PATH):
        """Initialize the Reader class.

        :param db_path: Path to the database file, defaults to DB_PATH.
        """
        self.db_path = db_path

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def get_recording_id(self, recording: str | int) -> int:
        """Get the recording ID from the database.

        :param recording: Recording ID or original filename of the recording.
        :raises ValueError: If the recording ID or original filename is not found.
        :return: Recording ID.
        """
        if isinstance(recording, int) or recording.isdigit():
            # Check if the recording ID exists
            self.cursor.execute("SELECT _id FROM Recording WHERE _id = ?", (int(recording),))
            row = self.cursor.fetchone()
            if row:
                return int(row[0])
            else:
                raise ValueError(f"Recording with ID '{recording}' not found.")

        if isinstance(recording, str):
            self.cursor.execute(
                "SELECT _id FROM Recording WHERE original_file = ?",
                (recording,),
            )
            row = self.cursor.fetchone()
            if row:
                return int(row[0])
            else:
                raise ValueError(f"Recording with original file '{recording}' not found.")

        raise TypeError("Recording must be an integer or a string.")
