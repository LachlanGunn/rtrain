#!/usr/bin/env python3

import base64
import json
import os


def get_database_location():
    return os.environ.get('DB_PATH', 'U:/Source/rtrain/testdb.db')


__queries = {
    "create_job": """
        INSERT INTO Jobs
            (id, creation_time, modification_time, status, finished, job)
        VALUES 
            (?, datetime('now', 'unixepoch'), datetime('now', 'unixepoch'), 0.0, 0, ?)
        """,
    "get_next_job": """
        SELECT id, job FROM Jobs
        WHERE finished = 0
        ORDER BY creation_time
        LIMIT 1
        """,
    "get_status": """
        SELECT status, finished FROM Jobs WHERE id=?
    """,
    "update_status": """
        UPDATE Jobs SET
            status = ?, modification_time = datetime('now', 'unixepoch')
        WHERE id = ?
        """,
    "finish_job": """
        UPDATE Jobs SET
            finished = 1, job = ?, modification_time = datetime('now', 'unixepoch')
        WHERE id = ?
        """,
    "get_results": """
        SELECT job FROM Jobs WHERE finished = 1 AND id = ?
    """,
    "purge_jobs": """
        DELETE FROM Jobs WHERE finished = 1 AND modification_time < datetime('now','unixepoch')-60
    """
}


def _create_job_id():
    job_id = base64.b32encode(os.urandom(20)).lower()
    assert len(job_id) == 32
    return job_id


def create_new_job(training_request, db):
    job_id = _create_job_id()

    training_data = json.dumps(training_request)

    cursor = db.cursor()
    cursor.execute(__queries['create_job'], (job_id, training_data))
    cursor.close()
    db.commit()

    return job_id


def get_next_job(db):
    cursor = db.cursor()
    cursor.execute(__queries['get_next_job'])

    result = cursor.fetchone()
    if result is None:
        return None
    else:
        (job_id, training_data) = result
    cursor.close()
    return job_id, training_data


def get_status(job_id, db):
    cursor = db.cursor()
    cursor.execute(__queries['get_status'], (bytearray(job_id, 'utf-8'),))

    result = cursor.fetchone()
    if result is None:
        return None
    else:
        (status, finished) = result
    cursor.close()
    return status, finished


def get_results(job_id, db):
    cursor = db.cursor()
    cursor.execute(__queries['get_results'], (bytearray(job_id, 'utf-8'),))

    result = cursor.fetchone()
    if result is None:
        return None
    else:
        result = result
    cursor.close()
    return result


def update_status(job_id, percentage, db):
    cursor = db.cursor()
    cursor.execute(__queries['update_status'], (percentage, job_id))
    cursor.close()
    db.commit()


def finish_job(job_id, result, db):
    cursor = db.cursor()
    cursor.execute(__queries['finish_job'], (result, job_id))
    cursor.close()
    db.commit()


def purge_old_jobs(db):
    cursor = db.cursor()
    cursor.execute(__queries['purge_jobs'])
    cursor.close()
    db.commit()
