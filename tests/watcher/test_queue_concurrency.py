"""
Tests de concurrence — queue.py

Reproduit la panne du 3 août 2026 sur l'instance mrae : la suppression
simultanée de cinq fichiers déclenche cinq `threading.Timer` dans l'observer,
qui appellent tous `JobQueue.enqueue` sur la MEME instance, donc la même
connexion SQLite. Deux enqueue échouent (« database is locked » puis
« cannot commit transaction - SQL statements in progress »), la transaction
d'écriture reste ouverte, et plus aucun composant du watcher ne peut écrire
jusqu'au redémarrage du processus.

Ces tests échouent sur le code d'avant le correctif.
"""

import sqlite3
import threading
import time


from src.watcher.queue import JobQueue


def _run_threads(cibles) -> list[BaseException]:
    """Lance les callables en parallèle et retourne les exceptions levées."""
    erreurs: list[BaseException] = []
    verrou = threading.Lock()
    depart = threading.Event()

    def _envelopper(fn):
        def _inner():
            depart.wait()
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 — on veut tout capturer
                with verrou:
                    erreurs.append(exc)

        return _inner

    fils = [threading.Thread(target=_envelopper(c)) for c in cibles]
    for f in fils:
        f.start()
    depart.set()
    for f in fils:
        f.join(timeout=30)
    return erreurs


class TestEnqueueConcurrent:
    def test_enqueues_simultanes_sur_une_instance(self, queue: JobQueue) -> None:
        """Dix fils, une seule JobQueue : aucun enqueue ne doit échouer."""
        nb_fils = 10
        par_fil = 5

        def _travail(i: int):
            def _fn():
                for j in range(par_fil):
                    queue.enqueue(f"/data/doc_{i}_{j}.pdf", "delete")

            return _fn

        erreurs = _run_threads([_travail(i) for i in range(nb_fils)])

        assert not erreurs, f"enqueue a levé : {[str(e) for e in erreurs]}"
        assert queue.get_counts_by_status().get("pending", 0) == nb_fils * par_fil

    def test_la_connexion_ne_reste_pas_en_transaction(self, queue: JobQueue) -> None:
        """Après une rafale concurrente, aucune transaction ne doit rester ouverte.

        C'est la transaction laissée ouverte par un commit en échec qui retient
        le verrou d'écriture et bloque tout le processus.
        """

        def _travail(i: int):
            def _fn():
                queue.enqueue(f"/data/rafale_{i}.pdf", "delete")

            return _fn

        _run_threads([_travail(i) for i in range(10)])

        assert queue._conn.in_transaction is False


class TestPasDeVerrouGlobal:
    def test_une_autre_connexion_ecrit_toujours(self, db_path: str) -> None:
        """Une rafale concurrente ne doit pas verrouiller les autres composants.

        Le worker, le reconciler et l'API ouvrent chacun leur propre connexion.
        Après la rafale, une écriture depuis une connexion distincte doit passer
        en moins de cinq secondes (le busy_timeout des PRAGMA).
        """
        observer_queue = JobQueue(db_path=db_path)
        worker_queue = JobQueue(db_path=db_path)

        def _travail(i: int):
            def _fn():
                observer_queue.enqueue(f"/data/rafale_{i}.pdf", "delete")

            return _fn

        try:
            _run_threads([_travail(i) for i in range(10)])

            debut = time.monotonic()
            worker_queue.enqueue("/data/depuis_le_worker.pdf", "upsert")
            duree = time.monotonic() - debut

            assert duree < 5.0, f"écriture concurrente bloquée {duree:.1f}s"
        finally:
            observer_queue.close()
            worker_queue.close()

    def test_dequeue_reste_possible_apres_rafale(self, db_path: str) -> None:
        """Le worker doit continuer à sortir des jobs après une rafale d'enqueue."""
        observer_queue = JobQueue(db_path=db_path)
        worker_queue = JobQueue(db_path=db_path)

        def _travail(i: int):
            def _fn():
                observer_queue.enqueue(f"/data/rafale_{i}.pdf", "delete")

            return _fn

        try:
            _run_threads([_travail(i) for i in range(10)])

            sortis = 0
            for _ in range(10):
                if worker_queue.dequeue() is not None:
                    sortis += 1

            assert sortis == 10, f"seulement {sortis}/10 jobs sortis de la file"
        finally:
            observer_queue.close()
            worker_queue.close()


class TestDequeueConcurrent:
    def test_un_job_n_est_sorti_qu_une_fois(self, db_path: str) -> None:
        """Deux workers sur la même file ne doivent pas traiter le même job."""
        producteur = JobQueue(db_path=db_path)
        for i in range(20):
            producteur.enqueue(f"/data/job_{i}.pdf", "upsert")

        workers = [JobQueue(db_path=db_path) for _ in range(4)]
        vus: list[str] = []
        verrou = threading.Lock()

        def _travail(q: JobQueue):
            def _fn():
                while True:
                    job = q.dequeue()
                    if job is None:
                        return
                    with verrou:
                        vus.append(job.job_id)

            return _fn

        try:
            erreurs = _run_threads([_travail(q) for q in workers])
            assert not erreurs, f"dequeue a levé : {[str(e) for e in erreurs]}"
            assert len(vus) == len(set(vus)), "un job a été sorti deux fois"
            assert len(vus) == 20, f"{len(vus)}/20 jobs sortis"
        finally:
            producteur.close()
            for q in workers:
                q.close()


class TestRemonteeDesVerrous:
    def test_le_compteur_de_verrous_est_expose(self, db_path: str) -> None:
        """Un verrou subi doit être compté, pas avalé en silence.

        La panne est restée invisible parce que `dequeue` attrapait
        OperationalError, journalisait en DEBUG et retournait None : le statut
        du service affichait `worker: true` pendant que rien n'avançait.
        """
        q = JobQueue(db_path=db_path)
        try:
            assert q.consecutive_lock_errors == 0

            bloqueur = sqlite3.connect(db_path, timeout=0.1)
            bloqueur.execute("BEGIN EXCLUSIVE")
            try:
                q.dequeue()
            finally:
                bloqueur.rollback()
                bloqueur.close()

            assert q.consecutive_lock_errors >= 1

            q.dequeue()
            assert q.consecutive_lock_errors == 0
        finally:
            q.close()
