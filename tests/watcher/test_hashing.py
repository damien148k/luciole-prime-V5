"""
Tests unitaires — hashing.py

Vérifie : quick_hash, content_hash, wait_stable
"""

from pathlib import Path

import pytest

from src.watcher.hashing import content_hash, quick_hash, wait_stable


class TestQuickHash:
    def test_retourne_une_chaine(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("contenu")
        result = quick_hash(f)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_meme_fichier_hash_identique(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("contenu stable")
        h1 = quick_hash(f)
        h2 = quick_hash(f)
        assert h1 == h2

    def test_modification_mtime_change_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("v1")
        h1 = quick_hash(f)
        # Modifier le fichier change la taille → nouveau hash
        f.write_text("version modifiee avec plus de texte")
        h2 = quick_hash(f)
        assert h1 != h2

    def test_fichier_absent_leve_oserror(self, tmp_path: Path) -> None:
        f = tmp_path / "inexistant.txt"
        with pytest.raises(OSError):
            quick_hash(f)


class TestContentHash:
    def test_retourne_sha256_hexadecimal(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_bytes(b"contenu de test")
        result = content_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_deterministe(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_bytes(b"contenu deterministe")
        h1 = content_hash(f)
        h2 = content_hash(f)
        assert h1 == h2

    def test_contenu_different_hash_different(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"contenu A")
        f2.write_bytes(b"contenu B")
        assert content_hash(f1) != content_hash(f2)

    def test_contenu_identique_hash_identique(self, tmp_path: Path) -> None:
        f1 = tmp_path / "copie1.txt"
        f2 = tmp_path / "copie2.txt"
        data = b"meme contenu binaire"
        f1.write_bytes(data)
        f2.write_bytes(data)
        assert content_hash(f1) == content_hash(f2)

    def test_fichier_vide(self, tmp_path: Path) -> None:
        f = tmp_path / "vide.txt"
        f.write_bytes(b"")
        result = content_hash(f)
        # SHA-256 de la chaîne vide
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_gros_fichier_par_blocs(self, tmp_path: Path) -> None:
        f = tmp_path / "gros.bin"
        # Fichier de 20 Mo (> CONTENT_HASH_BLOCK_SIZE de 8 Mo)
        data = b"x" * (20 * 1024 * 1024)
        f.write_bytes(data)
        result = content_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64


class TestWaitStable:
    def test_fichier_stable_retourne_true(self, tmp_path: Path) -> None:
        f = tmp_path / "stable.txt"
        f.write_text("contenu complet")
        result = wait_stable(f, checks=2, interval=0.05)
        assert result is True

    def test_fichier_absent_retourne_false(self, tmp_path: Path) -> None:
        f = tmp_path / "inexistant.txt"
        result = wait_stable(f, checks=2, interval=0.05)
        assert result is False

    def test_fichier_vide_retourne_false(self, tmp_path: Path) -> None:
        f = tmp_path / "vide.txt"
        f.write_bytes(b"")
        result = wait_stable(f, checks=2, interval=0.05)
        assert result is False
