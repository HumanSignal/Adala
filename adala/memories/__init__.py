import sqlite3

if sqlite3.sqlite_version_info < (3, 35, 0):
    # In Colab, hotswap to pysqlite-binary if it's too old
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pysqlite3-binary"]
    )
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from .file_memory import FileMemory
from .base import Memory

