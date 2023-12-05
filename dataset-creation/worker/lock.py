import os
import time
from db import DB


def acquire_lock(path):
    path = os.path.abspath(path)
    while True:
        sql = "INSERT INTO file_locks(`path`, `locked`)" \
              "VALUES(%s, 1) ON DUPLICATE KEY UPDATE " \
              "path=VALUES(path), locked=VALUES(locked)"
        _, rows = DB.fetch_one_with_row_count(sql, (path,))
        if rows:
            return
        else:
            DB.disconnect()
            time.sleep(10)


def release_lock(path):
    path = os.path.abspath(path)
    sql = "UPDATE file_locks SET locked=0 WHERE `path`=%s"
    DB.execute(sql, (path,))
