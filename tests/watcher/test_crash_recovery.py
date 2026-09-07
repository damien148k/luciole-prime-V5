"""
Tests d'intégration — Reprise après crash (recover_in_progress).

Simule un arrêt brutal du worker pendant le traitement d'un job
et vérifie que les jobs sont correctement récupérés au redémarrage.
"""

from pathlib import Path


from src.watcher.db import init_db
from src.watcher.models import JobStatus
from src.watcher.queue import JobQueue


class TestCrashRecovery:
    def test_jobs_in_progress_remis_pending_apres_crash(
        self, db_path: str
    ) -> None:
        """
        Simule un crash : des jobs sont en 'in_progress' dans la DB.
        Au redémarrage (recover_in_progress), ils doivent être remis en 'pending'.
        """
        queue = JobQueue(db_path=db_path)

        # Enqueue et démarrer 3 jobs (simule un worker qui les prend)
        job_ids = []
        for i in range(3):
            jid = queue.enqueue(f"/data/doc{i}.pdf", "upsert")
            job_ids.append(jid)

        # Démarrer les jobs (ils passent en in_progress)
        dequeued = []
        for _ in range(3):
            j = queue.dequeue()
            if j:
                dequeued.append(j)

        assert len(dequeued) == 3
        assert all(j.status == JobStatus.IN_PROGRESS.value for j in dequeued)

        # Simuler le crash — fermer la connexion sans compléter les jobs
        queue.close()

        # Redémarrage — nouvelle instance (nouveau processus)
        new_queue = JobQueue(db_path=db_path)
        recovered = new_queue.recover_in_progress()

        assert recovered == 3

        # Les jobs doivent maintenant être traitables
        pending = new_queue.get_pending_count()
        assert pending == 3

        new_queue.close()

    def test_jobs_completed_non_affectes_par_recover(
        self, db_path: str
    ) -> None:
        """
        Les jobs déjà 'completed' ne doivent pas être réinitialisés.
        """
        queue = JobQueue(db_path=db_path)

        jid = queue.enqueue("/data/ok.pdf", "upsert")
        j = queue.dequeue()
        queue.complete(j.job_id)  # Terminé proprement

        queue.close()

        new_queue = JobQueue(db_path=db_path)
        recovered = new_queue.recover_in_progress()
        assert recovered == 0

        completed = new_queue.get_job(jid)
        assert completed.status == JobStatus.COMPLETED.value

        new_queue.close()

    def test_db_persiste_entre_connexions(self, db_path: str) -> None:
        """
        Vérifie que les données persistent correctement sur disque.
        """
        # Session 1 : créer des données
        q1 = JobQueue(db_path=db_path)
        job_id = q1.enqueue("/data/persistant.pdf", "upsert")
        q1.close()

        # Session 2 : retrouver les données
        q2 = JobQueue(db_path=db_path)
        job = q2.get_job(job_id)
        assert job is not None
        assert job.file_path == "/data/persistant.pdf"
        q2.close()

    def test_schema_cree_si_db_absente(self, tmp_path: Path) -> None:
        """
        La base est créée et le schéma appliqué si le fichier n'existe pas.
        """
        new_db = str(tmp_path / "nouveau" / "watcher.db")
        conn = init_db(new_db)
        assert Path(new_db).exists()

        # Vérifier que les tables existent
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}

        assert "documents" in table_names
        assert "jobs" in table_names
        assert "watcher_events" in table_names
        assert "audit_errors" in table_names
        conn.close()

    def test_pragma_wal_actif(self, db_path: str) -> None:
        """Vérifie que le mode WAL est correctement appliqué."""
        from src.watcher.db import get_connection

        conn = get_connection(db_path)
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        conn.close()

    def test_pragma_foreign_keys_actif(self, db_path: str) -> None:
        """Vérifie que les foreign keys sont activées."""
        from src.watcher.db import get_connection

        conn = get_connection(db_path)
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1
        conn.close()
