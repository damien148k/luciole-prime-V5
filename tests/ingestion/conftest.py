"""
Fixtures partagées pour les tests de pagination (parsers.py, chunker.py).
"""

import sys
from pathlib import Path

import pytest

# Assurer que les imports src.* fonctionnent depuis le répertoire rag-system
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pymupdf


@pytest.fixture
def pdf_multi_pages(tmp_path: Path) -> Path:
    """
    Construit à la volée un PDF de 3 pages avec pymupdf, chaque page
    portant un texte distinct et identifiable.
    """
    chemin = tmp_path / "document_multi_pages.pdf"
    doc = pymupdf.open()
    textes = [
        "Contenu de la premiere page du document de test pour la pagination.",
        "Contenu de la deuxieme page, distinct du reste, pour verifier le decoupage.",
        "Contenu de la troisieme et derniere page qui termine le document.",
    ]
    for texte in textes:
        page = doc.new_page()
        page.insert_text((72, 72), texte, fontsize=11)
    doc.save(str(chemin))
    doc.close()
    return chemin


@pytest.fixture
def pdf_phrase_continue(tmp_path: Path) -> Path:
    """
    Construit un PDF de 3 pages dont le texte forme une seule suite de
    phrases continues, sans coupure nette a chaque frontiere de page :
    utile pour forcer un fragment de chunker a cheval sur deux pages.
    """
    chemin = tmp_path / "document_phrase_continue.pdf"
    doc = pymupdf.open()
    textes = [
        "Ceci est une phrase qui commence sur la premiere page et continue.",
        "Elle se termine ici sur la deuxieme page du document test.",
        "Puis une derniere phrase clot le document sur la troisieme page.",
    ]
    for texte in textes:
        page = doc.new_page()
        page.insert_text((72, 72), texte, fontsize=11)
    doc.save(str(chemin))
    doc.close()
    return chemin


@pytest.fixture
def pdf_page_vide(tmp_path: Path) -> Path:
    """
    Construit un PDF de 3 pages dont la page du milieu est vide, pour
    verifier que page_spans reste coherente en presence d'une page sans
    texte extractible.
    """
    chemin = tmp_path / "document_page_vide.pdf"
    doc = pymupdf.open()
    doc.new_page()  # page 1 : vide
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Seule la deuxieme page contient du texte.", fontsize=11)
    doc.new_page()  # page 3 : vide
    doc.save(str(chemin))
    doc.close()
    return chemin
