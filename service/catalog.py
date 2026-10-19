"""单机事务登记库；每次操作使用独立连接，支持HTTP和工作进程。"""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from .errors import APIError
from .storage import now


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Catalog:
    def __init__(self, root):
        self.path = Path(root)/'registry.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS defaults (
                    name TEXT PRIMARY KEY, value TEXT NOT NULL, activated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS idempotency (
                    scope TEXT NOT NULL, key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    expires REAL NOT NULL, result TEXT,
                    PRIMARY KEY(scope,key));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA busy_timeout=30000')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def put(self, kind, identifier, body, only_new=False):
        with self.connect() as db:
            sql = 'INSERT OR IGNORE' if only_new else 'INSERT OR REPLACE'
            db.execute(sql+' INTO records VALUES (?,?,?)', (kind, identifier, encode(body)))

    def get(self, kind, identifier):
        with self.connect() as db:
            row = db.execute('SELECT body FROM records WHERE kind=? AND id=?', (kind, identifier)).fetchone()
        if row is None:
            raise APIError(404, 'NOT_FOUND', '指定的记录不存在')
        return json.loads(row[0])

    def list(self, kind):
        with self.connect() as db:
            rows = db.execute('SELECT id,body FROM records WHERE kind=?', (kind,)).fetchall()
        return [json.loads(body) for identifier, body in sorted(rows, reverse=True)]

    def update_job(self, identifier, patch):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT body FROM records WHERE kind='job' AND id=?", (identifier,)).fetchone()
            if row is None:
                raise APIError(404, 'NOT_FOUND', '任务不存在')
            body = json.loads(row[0])
            body.update(patch(body) if callable(patch) else patch)
            db.execute("UPDATE records SET body=? WHERE kind='job' AND id=?", (encode(body), identifier))
            return body

    def default(self, name):
        with self.connect() as db:
            row = db.execute('SELECT value,activated_at FROM defaults WHERE name=?', (name,)).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def set_defaults(self, values, expected=None, only_missing=False):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = {key: db.execute('SELECT value,activated_at FROM defaults WHERE name=?', (key,)).fetchone()
                   for key in set(values) | set(expected or {})}
            for key, value in (expected or {}).items():
                if value is not None and (old[key][0] if old[key] else None) != value:
                    raise APIError(409, 'VERSION_CONFLICT', '默认版本已改变，请刷新后重试')
            changed = False
            stamp = now()
            for key, value in values.items():
                if only_missing and old[key]:
                    continue
                if not old[key] or old[key][0] != value:
                    db.execute('INSERT OR REPLACE INTO defaults VALUES (?,?,?)', (key, value, stamp))
                    changed = True
            return {key: old[key][0] if old[key] else None for key in old}, changed

    def reserve(self, scope, key, fingerprint, timestamp, retention):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM idempotency WHERE expires<?', (timestamp,))
            row = db.execute('SELECT fingerprint,result FROM idempotency WHERE scope=? AND key=?', (scope,key)).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise APIError(409, 'VERSION_CONFLICT', '同一Idempotency-Key对应了不同请求')
                if row[1] is None:
                    raise APIError(409, 'INVALID_STATE', '原请求仍在处理或执行状态待确认，请稍后查询记录')
                return json.loads(row[1])
            db.execute('INSERT INTO idempotency VALUES (?,?,?,?,NULL)', (scope,key,fingerprint,timestamp+retention))
        return None

    def finish(self, scope, key, result):
        with self.connect() as db:
            db.execute('UPDATE idempotency SET result=? WHERE scope=? AND key=?', (encode(result),scope,key))

