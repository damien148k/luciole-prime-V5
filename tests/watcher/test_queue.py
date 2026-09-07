"""
Tests unitaires — queue.py

Vérifie : enqueue, dequeue, complete, fail, mark_dead, retry, recover
"""



from src.watcher.models import JobStatus
from src.watcher.queue import JobQueue


class TestEnqueue:
    def test_enqueue_retourne_job_id(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/doc.pdf", "upsert")
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID

    def test_job_est_pending_apres_enqueue(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/doc.pdf", "upsert")
        job = queue.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.PENDING.value

    def test_enqueue_duplique_remplace_le_precedent(self, queue: JobQueue) -> None:
        # Deux upserts pour le même chemin → le premier est annulé
        id1 = queue.enqueue("/data/doc.pdf", "upsert")
        id2 = queue.enqueue("/data/doc.pdf", "upsert")
        # Le premier job doit être supprimé (dequeue ne le retourne plus)
        job1 = queue.get_job(id1)
        assert job1 is None  # annulé par le second enqueue
        job2 = queue.get_job(id2)
        assert job2 is not None
        assert job2.status == JobStatus.PENDING.value

    def test_enqueue_actions_differentes_coexistent(self, queue: JobQueue) -> None:
        id_upsert = queue.enqueue("/data/doc.pdf", "upsert")
        id_delete = queue.enqueue("/data/doc.pdf", "delete")
        assert queue.get_job(id_upsert) is not None
        assert queue.get_job(id_delete) is not None

    def test_priorite_haute_traitee_en_premier(self, queue: JobQueue) -> None:
        queue.enqueue("/data/normal.pdf", "upsert", priority=0)
        queue.enqueue("/data/urgent.pdf", "upsert", priority=1)
        job = queue.dequeue()
        assert job is not None
        assert job.file_path == "/data/urgent.pdf"


class TestDequeue:
    def test_dequeue_retourne_none_si_vide(self, queue: JobQueue) -> None:
        assert queue.dequeue() is None

    def test_dequeue_retourne_job_pending(self, queue: JobQueue) -> None:
        queue.enqueue("/data/doc.pdf", "upsert")
        job = queue.dequeue()
        assert job is not None
        assert job.status == JobStatus.IN_PROGRESS.value

    def test_dequeue_ne_retourne_pas_deux_fois_le_meme_job(self, queue: JobQueue) -> None:
        queue.enqueue("/data/doc.pdf", "upsert")
        job1 = queue.dequeue()
        job2 = queue.dequeue()
        assert job1 is not None
        assert job2 is None  # Le premier est in_progress, pas repris

    def test_dequeue_respecte_next_retry_at(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/doc.pdf", "upsert")
        # Simuler un job en attente de retry dans le futur
        from datetime import datetime, timedelta
        future = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        queue._conn.execute(
            "UPDATE jobs SET status='failed', next_retry_at=? WHERE job_id=?",
            (future, job_id),
        )
        queue._conn.commit()
        job = queue.dequeue()
        assert job is None  # Pas encore prêt


class TestComplete:
    def test_complete_marque_job_completed(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/doc.pdf", "upsert")
        job = queue.dequeue()
        queue.complete(job.job_id)
        updated = queue.get_job(job.job_id)
        assert updated.status == JobStatus.COMPLETED.value
        assert updated.completed_at is not None


class TestFail:
    def test_fail_marque_job_failed(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/doc.pdf", "upsert")
        job = queue.dequeue()
        queue.fail(job.job_id, "Erreur test")
        updated = queue.get_job(job.job_id)
        assert updated.status == JobStatus.FAILED.value
        assert "Erreur test" in updated.error_message
        assert updated.next_retry_at is not None

    def test_fail_passe_dead_apres_max_attempts(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/doc.pdf", "upsert")

        # Simuler max_attempts tentatives déjà effectuées
        queue._conn.execute(
            "UPDATE jobs SET attempts = max_attempts WHERE job_id = ?",
            (job_id,),
        )
        queue._conn.commit()

        job = queue.dequeue()
        queue.fail(job.job_id, "Trop d'erreurs")
        updated = queue.get_job(job.job_id)
        assert updated.status == JobStatus.DEAD.value


class TestRetry:
    def test_retry_remet_job_pending(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/doc.pdf", "upsert")
        job = queue.dequeue()
        queue.fail(job.job_id, "erreur")
        queue.retry_job(job.job_id)
        updated = queue.get_job(job.job_id)
        assert updated.status == JobStatus.PENDING.value
        assert updated.next_retry_at is None
        assert updated.error_message is None

    def test_retry_all_failed(self, queue: JobQueue) -> None:
        for i in range(3):
            jid = queue.enqueue(f"/data/doc{i}.pdf", "upsert")
            j = queue.dequeue()
            queue.fail(j.job_id, "err")
        count = queue.retry_all_failed()
        assert count == 3


class TestRecoverInProgress:
    def test_jobs_in_progress_remis_en_pending(self, queue: JobQueue) -> None:
        for _ in range(2):
            jid = queue.enqueue("/data/doc.pdf", "upsert")
            queue.dequeue()  # met en in_progress

        recovered = queue.recover_in_progress()
        assert recovered == 2

        pending_count = queue.get_pending_count()
        assert pending_count == 2


class TestGetPendingCount:
    def test_compte_pending_et_failed(self, queue: JobQueue) -> None:
        queue.enqueue("/data/a.pdf", "upsert")
        queue.enqueue("/data/b.pdf", "upsert")
        assert queue.get_pending_count() == 2

    def test_completed_non_compte(self, queue: JobQueue) -> None:
        job_id = queue.enqueue("/data/a.pdf", "upsert")
        job = queue.dequeue()
        queue.complete(job.job_id)
        assert queue.get_pending_count() == 0
