PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS network_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ping REAL NOT NULL CHECK(ping >= 0),
    jitter REAL NOT NULL CHECK(jitter >= 0),
    download_speed REAL NOT NULL CHECK(download_speed >= 0),
    upload_speed REAL NOT NULL CHECK(upload_speed >= 0),
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_network_logs_timestamp
ON network_logs(timestamp);

CREATE TABLE IF NOT EXISTS network_flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_ip TEXT NOT NULL,
    dst_ip TEXT NOT NULL,
    src_port INTEGER,
    dst_port INTEGER,
    protocol TEXT,
    bytes INTEGER NOT NULL CHECK(bytes >= 0),
    packets INTEGER CHECK(packets IS NULL OR packets >= 0),
    timestamp TEXT NOT NULL,
    device TEXT
);
CREATE INDEX IF NOT EXISTS idx_network_flows_timestamp
ON network_flows(timestamp);
