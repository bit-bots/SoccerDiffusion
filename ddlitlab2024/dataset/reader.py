import sqlite3

from ddlitlab2024 import DB_PATH


def get_connection(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    return conn
