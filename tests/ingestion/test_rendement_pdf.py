"""
Tests unitaires — garde-fou de rendement PDF et en-têtes de tableaux Markdown.

Vérifie :
  - le garde-fou de rendement de PDFParser ré-extrait en texte simple les
    pages dont le markdown pymupdf4llm est anormalement pauvre (pertes
    silencieuses constatées sur des pages A3 paysage à mise en page
    graphique dense, ~35 % du texte récupéré au global sans le greffon
    pymupdf-layout)
  - les pages quasi vides (< 200 chars de référence) ne sont pas re-extraites
  - le garde-fou est désactivable (min_yield_ratio=0)
  - le chunker ré-injecte l'en-tête d'un tableau Markdown dans les fragments
    qui commencent au milieu du tableau
  - _split_oversized coupe de préférence sur une frontière de ligne
"""

from pathlib import Path

import pytest

from src.ingestion.chunker import Chunker
from src.ingestion.parsers import PDFParser
import src.ingestion.parsers as parsers_module


def document_pour_chunker(content: str, metadata: dict = None) -> dict:
    base_metadata = {"file_name": "test.pdf", "file_path": "/tmp/test.pdf", "type": "pdf"}
    if metadata:
        base_metadata.update(metadata)
    return {"content": content, "metadata": base_metadata}


@pytest.fixture
def pdf_deux_pages_denses(tmp_path: Path) -> Path:
    """PDF de 2 pages avec un texte suffisamment long pour dépasser le
    seuil absolu du garde-fou (200 chars de référence). insert_textbox
    (et non insert_text) pour que le texte soit réellement inséré sur
    plusieurs lignes au lieu d'être tronqué au bord de page."""
    import pymupdf

    chemin = tmp_path / "document_dense.pdf"
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 520, 760),
            f"Page {i + 1}. " + "Texte dense de la page. " * 60,
            fontsize=11,
        )
    doc.save(str(chemin))
    doc.close()
    return chemin


class TestGardeFouRendement:
    """Le garde-fou compare le rendement pymupdf4llm de chaque page à
    l'extraction simple de la même page et ré-extrait les pages perdues."""

    @staticmethod
    def _espion_reprises(monkeypatch) -> list:
        """Enrobe le garde-fou pour compter les reprises effectives."""
        reprises = []
        original = PDFParser._appliquer_garde_fou_rendement

        def espion(self, doc_ref, p, text_md):
            texte, repris = original(self, doc_ref, p, text_md)
            reprises.append(repris)
            return texte, repris

        monkeypatch.setattr(PDFParser, "_appliquer_garde_fou_rendement", espion)
        return reprises

    def test_page_perdue_est_reextraite_en_texte_simple(
        self, pdf_deux_pages_denses: Path, monkeypatch
    ) -> None:
        """Simule la perte silencieuse : pymupdf4llm ne rend qu'un titre de
        quelques caractères pour chaque page."""

        def to_markdown_appauvri(file_path, pages=None, page_chunks=False, **kwargs):
            return [{"text": "Titre seul\n", "metadata": {}} for _ in pages]

        monkeypatch.setattr(
            parsers_module.pymupdf4llm, "to_markdown", to_markdown_appauvri
        )
        reprises = self._espion_reprises(monkeypatch)

        parser = PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_deux_pages_denses))
        content = result["content"]

        # Les pages ont été re-extraites : le texte dense est présent.
        assert "Texte dense de la page" in content
        assert reprises and all(reprises), "chaque page doit avoir été reprise"
        # page_spans reste cohérente : 2 pages, offsets ordonnés.
        page_spans = result["metadata"]["page_spans"]
        assert [p for _, _, p in page_spans] == [1, 2]
        for debut, fin, _ in page_spans:
            assert content[debut:fin].strip()

    def test_page_quasi_vide_pas_de_reextraction(
        self, pdf_multi_pages: Path, monkeypatch
    ) -> None:
        """Les pages dont la référence fait moins de 200 chars ne sont pas
        touchées : un écart absolu minime y est normal (titres, pieds de
        page, sauts de ligne). Un garde-fou à 99 % déclencherait sur le
        moindre écart si le seuil absolu n'existait pas."""
        reprises = self._espion_reprises(monkeypatch)

        parser = PDFParser(enable_ocr=False, min_yield_ratio=0.99)
        result = parser.parse(str(pdf_multi_pages))

        assert "Contenu de la premiere page" in result["content"]
        assert reprises, "le garde-fou doit avoir été évalué"
        assert not any(reprises), "aucune page ne doit être reprise sous le seuil absolu"

    def test_garde_fou_desactivable(
        self, pdf_deux_pages_denses: Path, monkeypatch
    ) -> None:
        """min_yield_ratio=0 désactive le garde-fou : le markdown pauvre
        est conservé tel quel."""

        def to_markdown_appauvri(file_path, pages=None, page_chunks=False, **kwargs):
            return [{"text": "Titre seul\n", "metadata": {}} for _ in pages]

        monkeypatch.setattr(
            parsers_module.pymupdf4llm, "to_markdown", to_markdown_appauvri
        )

        parser = PDFParser(enable_ocr=False, min_yield_ratio=0)
        result = parser.parse(str(pdf_deux_pages_denses))

        assert "Texte dense de la page" not in result["content"]
        assert "Titre seul" in result["content"]


def _grand_tableau(nb_lignes: int = 40) -> str:
    """Construit un tableau Markdown assez long pour être découpé en
    plusieurs fragments avec chunk_size=300."""
    lignes = [
        "|N|Commune|Monument|Protection|Aire|",
        "|---|---|---|---|---|",
    ]
    for i in range(nb_lignes):
        lignes.append(
            f"|{i}|Commune-{i:02d}|Monument historique numero {i}|Classe|AEE|"
        )
    return "\n".join(lignes)


class TestEntetesTableauxMarkdown:
    def test_detection_bloc_et_entete(self) -> None:
        content = "Introduction.\n\n" + _grand_tableau(5) + "\n\nConclusion.\n"
        tableaux = Chunker._detecter_tableaux(content)

        assert len(tableaux) == 1
        tableau = tableaux[0]
        assert content[tableau["debut"]:].startswith("|N|Commune|")
        assert tableau["entete"].split("\n")[0] == "|N|Commune|Monument|Protection|Aire|"
        assert tableau["entete"].split("\n")[1] == "|---|---|---|---|---|"

    def test_entete_reinjectee_dans_les_fragments_suivants(self) -> None:
        content = _grand_tableau(40)
        chunker = Chunker(
            chunk_size=300, chunk_overlap=0, strategy="sentence",
            include_file_context=False,
        )
        chunks = chunker.chunk(document_pour_chunker(content))

        assert len(chunks) > 2, "le tableau doit être découpé en plusieurs fragments"
        for i, chunk in enumerate(chunks):
            premiere_ligne = chunk.text.split("\n")[0]
            assert premiere_ligne.startswith("|N|Commune|"), (
                f"fragment {i} sans en-tête de tableau : {premiere_ligne!r}"
            )
            # Chaque ligne de fragment est une ligne de tableau complète
            for ligne in chunk.text.split("\n"):
                if ligne.startswith("|") and ligne != "|---|---|---|---|---|":
                    assert ligne.rstrip().endswith("|"), (
                        f"ligne de tableau tronquée dans fragment {i} : {ligne!r}"
                    )

    def test_pas_de_reinjection_hors_tableau(self) -> None:
        content = (
            "Premiere phrase du document. Deuxieme phrase du document. "
            "Troisieme phrase un peu plus longue pour depasser. "
        ) * 10
        chunker = Chunker(
            chunk_size=120, chunk_overlap=0, strategy="sentence",
            include_file_context=False,
        )
        chunks = chunker.chunk(document_pour_chunker(content))
        assert len(chunks) > 1
        for chunk in chunks:
            assert not chunk.text.startswith("|")

    def test_split_oversized_coupe_sur_frontiere_de_ligne(self) -> None:
        chunker = Chunker()
        lignes = [f"|{i}|valeur de la cellule numero {i}|autre cellule {i}|" for i in range(30)]
        texte = "\n".join(lignes)
        morceaux = chunker._split_oversized(texte, limit=300, overlap=0)

        assert len(morceaux) > 1
        for morceau in morceaux:
            for ligne in morceau.split("\n"):
                assert ligne.rstrip().endswith("|") or not ligne.startswith("|"), (
                    f"ligne de tableau coupée en deux : {ligne!r}"
                )
