#!/usr/bin/env python3
"""Database operations for rtraind."""

import base64
import datetime
import json
import os
import hashlib
import sqlalchemy

import rtrain.server_utils.model as model


def _create_job_id():
    """Create a new job ID."""
    job_id = str(base64.b32encode(os.urandom(20)).lower(), 'ascii')
    return job_id


def create_new_job(training_request, session):
    """Insert a new job into the database."""
    job_id = _create_job_id()

    training_data = json.dumps(training_request)

    digest = hashlib.sha256()
    digest.update(training_data.encode())

    new_job = model.Job(id=job_id, status=0, finished=0, job_type='train')
    new_training = model.TrainingJob(
        job_id=job_id,
        training_job=training_data.encode('utf8'),
        job_checksum=str(base64.b16encode(digest.digest()), 'ascii'))
    session.add(new_job)
    session.add(new_training)
    session.commit()

    return job_id


def get_next_job(session):
    """Get the next unfinished job from the database."""
    return session.query(model.Job).filter_by(
        finished=0).order_by(model.Job.creation_time).first()


def get_status(job_id, session):
    """Get the status of a particular job from the database."""
    return session.query(model.Job.finished, model.Job.status).filter_by(
        id=job_id).first()


def get_results(job_id, session):
    """Get the results of a training job from the database."""
    job = session.query(model.Job).filter_by(id=job_id).first()
    if not job.training_results:
        return None
    return str(job.training_results[0].result, 'utf8')


def update_status(job_id, percentage, session):
    """Update the status of a job in the database."""
    job = session.query(model.Job).filter_by(id=job_id).first()
    job.status = percentage
    session.commit()


def finish_job(job_id, result, session):
    """Mark a training job as finished in the database."""
    job = session.query(model.Job).filter_by(id=job_id).first()
    job.finished = 1

    training_result = model.TrainingResult(
        job_id=job.id, result=result.encode('utf8'))
    session.add(training_result)
    session.commit()


def purge_old_jobs(session):
    """Purge jobs older than one minute from the database."""
    cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
    session.query(model.Job).filter(model.Job.modification_time < cutoff_time,
                                    model.Job.finished != 0).delete()
    session.commit()
