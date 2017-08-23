#!/usr/bin/env python

import pytest
import datetime

import rtrain.server_utils.model as model
import rtrain.server_utils.model.database_operations as ops


@pytest.fixture
def session():
    import sqlalchemy
    import sqlalchemy.orm

    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    model.Base.metadata.create_all(engine)
    return sqlalchemy.orm.Session(bind=engine)


def test_create_new_job(session):
    job_id = ops.create_new_job(['foobarbaz'], session)

    results = session.query(model.Job)
    assert results.count() == 1

    job = results.first()
    assert job.id == job_id
    assert job.status == 0.0
    assert job.finished == 0

    assert abs(job.creation_time - datetime.datetime.utcnow()) < datetime.timedelta(minutes=1)

    assert len(job.training_results) == 0
    assert len(job.training_jobs) == 1
    assert job.training_jobs[0].training_job == '["foobarbaz"]'


def test_update_status(session):
    job_id = ops.create_new_job([], session)
    ops.update_status(job_id, 3.14159, session)

    result = session.query(model.Job).first()
    assert result.status == pytest.approx(3.14159)
    assert result.finished == 0


def test_get_status(session):
    job_id = ops.create_new_job([], session)
    job = session.query(model.Job).first()
    job.status = 3.14159
    session.commit()

    status = ops.get_status(job_id, session)
    assert status.status == pytest.approx(3.14159)


def test_finish(session):
    job_id = ops.create_new_job([], session)
    ops.finish_job(job_id, 'result', session)

    result = session.query(model.Job).first()
    assert result.finished == 1
    assert len(result.training_results) == 1
    assert result.training_results[0].result == 'result'


def test_get_results(session):
    job_id = ops.create_new_job([], session)
    ops.finish_job(job_id, 'result', session)

    result = ops.get_results(job_id, session)
    assert result == 'result'


def test_purge(session):
    job_id_1 = ops.create_new_job([], session)
    ops.finish_job(job_id_1, 'result', session)

    job_id_2 = ops.create_new_job([], session)
    ops.finish_job(job_id_2, 'result', session)
    job_2 = session.query(model.Job).filter_by(id=job_id_2).first()
    job_2.modification_time = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
    session.commit()

    job_id_3 = ops.create_new_job([], session)
    job_2 = session.query(model.Job).filter_by(id=job_id_3).first()
    job_2.modification_time = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
    session.commit()

    job_id_4 = ops.create_new_job([], session)
    ops.finish_job(job_id_4, 'result', session)
    job_4 = session.query(model.Job).filter_by(id=job_id_4).first()
    job_4.modification_time = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    session.commit()

    ops.purge_old_jobs(session)

    jobs = session.query(model.Job).order_by(model.Job.modification_time).all()

    assert len(jobs) == 3
    assert jobs[0].id == job_id_3
    assert jobs[1].id == job_id_1
    assert jobs[2].id == job_id_4


def test_get_next_job(session):
    job_id_1 = ops.create_new_job([], session)
    ops.finish_job(job_id_1, 'result', session)

    job_id_2 = ops.create_new_job([], session)
    ops.finish_job(job_id_2, 'result', session)
    job_2 = session.query(model.Job).filter_by(id=job_id_2).first()
    job_2.creation_time = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
    session.commit()

    job_id_3 = ops.create_new_job([], session)
    job_2 = session.query(model.Job).filter_by(id=job_id_3).first()
    job_2.creation_time = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
    session.commit()

    job_id_4 = ops.create_new_job([], session)
    ops.finish_job(job_id_4, 'result', session)
    job_4 = session.query(model.Job).filter_by(id=job_id_4).first()
    job_4.creation_time = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    session.commit()

    job = ops.get_next_job(session)
    assert job.id == job_id_3
