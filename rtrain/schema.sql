-- RTrain database schema.

DROP TABLE IF EXISTS Jobs;

CREATE TABLE Jobs (
    id                  CHARACTER(32) PRIMARY KEY, -- 132 bits in Base32
    creation_time       INTEGER,
    modification_time   INTEGER,
    status              REAL,
    finished            BOOLEAN,
    job                 TEXT
);