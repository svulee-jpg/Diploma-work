"""SQLite-хранилище прототипа. В продакшне заменяется на PostgreSQL без изменения логики."""
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(
    os.environ.get(
        "SMARTPOS_DB",
        Path(__file__).resolve().parent.parent / "data" / "smartpos.db",
    )
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    phone      TEXT,
    consent    INTEGER NOT NULL DEFAULT 0,   -- согласие на обработку биометрии (opt-in)
    bonus      INTEGER NOT NULL DEFAULT 0,
    prefs      TEXT NOT NULL DEFAULT '[]',   -- JSON-список предпочтений («Без сахара»)
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (      -- только обезличенные векторы, без фотографий
    customer_id INTEGER PRIMARY KEY REFERENCES customers(id),
    vec         BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,                -- drink | food
    price      INTEGER NOT NULL,
    stock      INTEGER NOT NULL DEFAULT 0,
    hours_left REAL                          -- часы до конца срока годности (NULL для напитков)
);
CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER REFERENCES customers(id),
    ts          TEXT NOT NULL,
    total       INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'done',     -- new | preparing | ready | done (старые чеки = done)
    source      TEXT NOT NULL DEFAULT 'barista',  -- kiosk | barista
    started_at  TEXT,
    ready_at    TEXT,
    done_at     TEXT
);
CREATE TABLE IF NOT EXISTS order_items (
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    qty        INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS rec_log (         -- журнал подсказок для оценки конверсии
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    customer_id INTEGER,
    product_id  INTEGER NOT NULL,
    probability REAL NOT NULL,
    accepted    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS scan_log (        -- результаты распознавания (без изображений и векторов)
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    matched INTEGER NOT NULL,
    score   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id, ts);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    # миграция для БД, созданных до появления очереди заказов
    cols = {r["name"] for r in con.execute("PRAGMA table_info(orders)")}
    for name, ddl in [("status", "TEXT NOT NULL DEFAULT 'done'"), ("source", "TEXT NOT NULL DEFAULT 'barista'"),
                      ("started_at", "TEXT"), ("ready_at", "TEXT"), ("done_at", "TEXT")]:
        if name not in cols:
            con.execute(f"ALTER TABLE orders ADD COLUMN {name} {ddl}")
    con.commit()
    return con
