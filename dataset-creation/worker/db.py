import backoff
import MySQLdb


class DB:
    conn = None

    @staticmethod
    # @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def connect():
        with open('../mysql/hostname') as file:
            hostname = file.read().strip()

        DB.conn = MySQLdb.connect(
            hostname,
            'xt86',
            'Robert999',
            'molm_dataset'
        )

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def fetch_all(sql, params=None, commit=True):
        cursor = DB.query(sql, params, commit)
        result = cursor.fetchall()
        if result is not None:
            return [dict(zip(list(zip(*cursor.description))[0], row)) for row in result]
        else:
            return None

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def fetch_one(sql, params=None, commit=True):
        cursor = DB.query(sql, params, commit)
        result = cursor.fetchone()
        if result is not None:
            return dict(zip(list(zip(*cursor.description))[0], result))
        else:
            return None

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def fetch_one_with_row_count(sql, params=None):
        cursor = DB.query(sql, params, commit=True)
        rows = cursor.rowcount
        result = cursor.fetchone()
        if result is not None:
            return dict(zip(list(zip(*cursor.description))[0], result)), rows
        else:
            return None, (rows if rows else 0)

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def query(sql, params=None, commit=True):
        if not DB.conn:
            DB.connect()

        args = []
        if params:
            args = [params]

        try:
            cursor = DB.conn.cursor()
            cursor.execute(sql, *args)
            if commit:
                DB.conn.commit()
        except (AttributeError, MySQLdb.OperationalError):
            DB.connect()

            cursor = DB.conn.cursor()
            cursor.execute(sql, *args)
            if commit:
                DB.conn.commit()

        return cursor

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def execute(sql, params=None, commit=True):
        cursor = DB.query(sql, params, commit)
        cursor.fetchone()
        return cursor

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def commit():
        DB.conn.commit()
        DB.disconnect()

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def rollback():
        DB.conn.rollback()

    @staticmethod
    def disconnect():
        if DB.conn is None:
            return

        try:
            DB.conn.close()
        except:
            pass
        finally:
            DB.conn = None
