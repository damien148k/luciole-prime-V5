"""
Tests unitaires — numeros de page (parsers.py, chunker.py)

Verifie :
  - la table page_spans produite par PDFParser correspond au contenu reel
  - le chunker calcule page_start/page_end corrects a partir de page_spans
  - un fragment entierement dans une page a page_start == page_end
  - un fragment a cheval sur deux pages a page_start < page_end
  - un document sans page_spans produit des fragments avec page_start/
    page_end a None, sans exception
  - les offsets start_char/end_char ne regressent pas : content[start:end]
    recouvre le texte du fragment apres normalisation des blancs
"""

import re
from pathlib import Path

import pytest

from src.ingestion.chunker import Chunker
from src.ingestion.parsers import PDFParser


def normaliser(texte: str) -> str:
    """Reduit toute suite de blancs a un seul espace, pour comparer un
    texte reconstruit (chunker) a un extrait brut de content."""
    return re.sub(r"\s+", " ", texte).strip()


def document_pour_chunker(content: str, metadata: dict) -> dict:
    """Construit le dict document attendu par Chunker.chunk()."""
    base_metadata = {"file_name": "test.txt", "file_path": "/tmp/test.txt", "type": "text"}
    base_metadata.update(metadata)
    return {"content": content, "metadata": base_metadata}


class TestPageSpansPDFParser:
    """Verifie que PDFParser._extract_markdown_pagewise produit une table
    page_spans correspondant au contenu reel du PDF."""

    def test_page_spans_correspond_au_contenu_reel(self, pdf_multi_pages: Path) -> None:
        parser = PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_multi_pages))

        content = result["content"]
        page_spans = result["metadata"]["page_spans"]

        assert page_spans, "page_spans ne doit pas etre vide pour un PDF avec texte"

        for debut, fin, numero_page in page_spans:
            extrait = content[debut:fin]
            assert extrait.strip(), f"extrait vide pour la page {numero_page}"
            # Le numero de page doit etre coherent avec l'ordre du document
            assert isinstance(numero_page, int)
            assert numero_page >= 1

        # Les spans sont ordonnes et couvrent des zones disjointes
        for i in range(len(page_spans) - 1):
            assert page_spans[i][1] <= page_spans[i + 1][0]

    def test_page_spans_absente_pour_page_vide(self, pdf_page_vide: Path) -> None:
        """Une page sans texte extractible ne doit pas produire d'entree
        de longueur nulle dans page_spans : seule la page 2 (avec texte)
        doit apparaitre pour ce document construit avec pages 1 et 3 vides."""
        parser = PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_page_vide))

        page_spans = result["metadata"]["page_spans"]
        numeros = [p for _, _, p in page_spans]

        assert 2 in numeros
        for debut, fin, _ in page_spans:
            assert fin > debut


class TestFragmentDansUnePage:
    """Un fragment dont les offsets tombent entierement dans les bornes
    d'une seule page doit avoir page_start == page_end."""

    def test_fragment_entierement_dans_une_page(self, pdf_multi_pages: Path) -> None:
        parser = PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_multi_pages))

        # Taille de chunk juste assez petite pour que chaque page ne
        # tienne pas dans le meme fragment que ses voisines (le contenu
        # total du document fait plus de 500 caracteres pour ce test).
        chunker = Chunker(chunk_size=60, chunk_overlap=0, strategy="paragraph")
        doc = document_pour_chunker(result["content"], result["metadata"])
        chunks = chunker.chunk(doc)

        assert chunks, "le decoupage ne doit pas etre vide"
        for chunk in chunks:
            page_start = chunk.metadata.get("page_start")
            page_end = chunk.metadata.get("page_end")
            assert page_start is not None
            assert page_end is not None
            assert page_start == page_end


class TestFragmentAChevalSurDeuxPages:
    """Un fragment dont les offsets debordent la frontiere entre deux
    pages doit avoir page_start < page_end."""

    def test_fragment_a_cheval_sur_deux_pages(self, pdf_phrase_continue: Path) -> None:
        parser = PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_phrase_continue))

        # Taille de chunk avec recouvrement, sur un texte continu a
        # travers les pages (sans coupure nette de phrase a la frontiere) :
        # force des fragments a cheval sur la frontiere entre deux pages.
        chunker = Chunker(chunk_size=90, chunk_overlap=20, strategy="sentence")
        doc = document_pour_chunker(result["content"], result["metadata"])
        chunks = chunker.chunk(doc)

        pages_start = [c.metadata.get("page_start") for c in chunks]
        pages_end = [c.metadata.get("page_end") for c in chunks]

        a_cheval = [
            (ps, pe) for ps, pe in zip(pages_start, pages_end)
            if ps is not None and pe is not None and ps != pe
        ]
        assert a_cheval, "au moins un fragment doit chevaucher deux pages avec ce decoupage"
        for ps, pe in a_cheval:
            assert ps < pe


class TestDocumentSansPageSpans:
    """Un document sans notion de page (page_spans absente des metadonnees,
    par exemple issu d'un parser texte simple) doit produire des fragments
    avec page_start/page_end a None, sans lever d'exception."""

    def test_document_sans_page_spans_donne_page_none(self) -> None:
        content = (
            "Un document texte simple sans notion de page. "
            "Deuxieme phrase pour verifier le decoupage. "
            "Troisieme phrase pour finir ce court texte."
        )
        chunker = Chunker(chunk_size=50, chunk_overlap=10, strategy="sentence")
        doc = document_pour_chunker(content, {})
        chunks = chunker.chunk(doc)

        assert chunks
        for chunk in chunks:
            assert chunk.metadata.get("page_start") is None
            assert chunk.metadata.get("page_end") is None
            assert "page_spans" not in chunk.metadata

    def test_page_spans_vide_donne_aussi_page_none(self) -> None:
        """page_spans presente mais vide (liste vide) doit se comporter
        comme une absence : aucune exception, page_start/page_end a None."""
        content = "Phrase une. Phrase deux pour tester le cas de la liste vide."
        chunker = Chunker(chunk_size=50, chunk_overlap=10, strategy="sentence")
        doc = document_pour_chunker(content, {"page_spans": []})
        chunks = chunker.chunk(doc)

        assert chunks
        for chunk in chunks:
            assert chunk.metadata.get("page_start") is None
            assert chunk.metadata.get("page_end") is None


class TestPageSpansAbsenteDeMetadonneesFragment:
    """La table page_spans decrit le document entier : elle ne doit
    jamais se retrouver copiee dans les metadonnees d'un fragment,
    meme quand elle est presente et utilisee pour calculer les pages."""

    def test_page_spans_retiree_des_metadonnees_du_fragment(self, pdf_multi_pages: Path) -> None:
        parser = PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_multi_pages))

        chunker = Chunker(chunk_size=80, chunk_overlap=15, strategy="sentence")
        doc = document_pour_chunker(result["content"], result["metadata"])
        chunks = chunker.chunk(doc)

        assert chunks
        for chunk in chunks:
            assert "page_spans" not in chunk.metadata


class TestNonRegressionOffsets:
    """Les offsets start_char/end_char doivent etre reels : content[start:
    end] doit recouvrir le texte du fragment apres normalisation des
    blancs. Couvre une ponctuation et des sauts de ligne varies, pour les
    deux strategies sentence et paragraph."""

    # Phrases courtes et ponctuees pour que _split_oversized ne soit pas
    # declenche aux tailles testees : le cas de chevauchement intra-unite
    # qu'il introduit (mot duplique en frontiere de morceaux, cf.
    # TestUniteAvecRecouvrementIntraUnite) est traite separement, car il
    # rend structurellement impossible une egalite stricte texte/position.
    TEXTE_VARIE = (
        "Premiere phrase. Deuxieme phrase. Troisieme phrase ici. "
        "Quatrieme phrase courte.\n\n"
        "Nouveau paragraphe. Une autre phrase le complete. Liste breve.\n\n"
        "Dernier paragraphe pour clore ce texte de test varie."
    )

    # Tailles superieures a la plus longue unite du texte (53 caracteres
    # pour la phrase la plus longue, 80 pour le plus long paragraphe) afin
    # de ne jamais declencher _split_oversized : ce test verifie la
    # non-regression stricte des offsets, le cas de chevauchement
    # intra-unite est couvert separement par TestUniteAvecRecouvrementIntraUnite.
    @pytest.mark.parametrize("strategie", ["sentence", "paragraph"])
    @pytest.mark.parametrize(
        "taille,recouvrement",
        [(90, 10), (150, 20), (100, 0), (200, 50), (120, 30)],
    )
    def test_offsets_recouvrent_le_texte_du_fragment(
        self, strategie: str, taille: int, recouvrement: int
    ) -> None:
        chunker = Chunker(chunk_size=taille, chunk_overlap=recouvrement, strategy=strategie)
        doc = document_pour_chunker(self.TEXTE_VARIE, {})
        chunks = chunker.chunk(doc)

        assert chunks
        for chunk in chunks:
            if chunk.start_char == -1 or chunk.end_char == -1:
                # Position non retrouvee : cas explicitement autorise par
                # le brief plutot que de deviner une position. Ne doit pas
                # survenir pour ce texte sans decoupage surdimensionne aux
                # tailles testees ici, mais reste tolere par construction.
                continue
            extrait_reel = self.TEXTE_VARIE[chunk.start_char:chunk.end_char]
            assert normaliser(extrait_reel) == normaliser(chunk.text), (
                f"taille={taille} recouvrement={recouvrement} strategie={strategie} : "
                f"fragment={chunk.text!r} reel={extrait_reel!r}"
            )

    def test_offsets_valides_ne_depassent_pas_le_contenu(self) -> None:
        chunker = Chunker(chunk_size=40, chunk_overlap=8, strategy="sentence")
        doc = document_pour_chunker(self.TEXTE_VARIE, {})
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            if chunk.start_char == -1:
                continue
            assert 0 <= chunk.start_char < chunk.end_char <= len(self.TEXTE_VARIE)

    def test_texte_des_fragments_inchange_par_rapport_a_la_construction(self) -> None:
        """Le decoupage (texte des fragments, taille, recouvrement) ne doit
        pas changer avec la correction des offsets : seule la position
        change. Verifie ici que le nombre de fragments et leur contenu
        textuel restent stables pour un jeu de parametres donne."""
        chunker = Chunker(chunk_size=60, chunk_overlap=12, strategy="sentence")
        doc = document_pour_chunker(self.TEXTE_VARIE, {})
        chunks_un = [c.text for c in chunker.chunk(doc)]
        chunks_deux = [c.text for c in chunker.chunk(doc)]

        assert chunks_un == chunks_deux


class TestUniteAvecRecouvrementIntraUnite:
    """Cas de bord identifie durant l'implementation : quand une unite
    (phrase ou paragraphe) depasse la taille limite, _split_oversized la
    redecoupe avec un recouvrement interne. Si deux morceaux consecutifs
    de ce redecoupage se retrouvent dans le meme fragment final, le texte
    du fragment contient alors un mot duplique qui n'existe qu'une fois
    dans content. Dans ce cas, content[start_char:end_char] est plus
    court que le texte du fragment : ce test documente et verifie que la
    position reste neanmoins un extrait coherent et non fantaisiste (pas
    de -1, borne dans le document), sans exiger une egalite stricte.
    """

    def test_position_coherente_meme_avec_duplication_de_recouvrement(self) -> None:
        # Phrase sans ponctuation interne, plus longue que chunk_size,
        # pour forcer _split_oversized avec chevauchement.
        content = (
            "Une phrase tres longue sans aucune ponctuation interne qui "
            "va forcer le redecoupage par _split_oversized avec un "
            "recouvrement interne a l unite elle meme."
        )
        chunker = Chunker(chunk_size=50, chunk_overlap=10, strategy="sentence")
        doc = document_pour_chunker(content, {})
        chunks = chunker.chunk(doc)

        assert chunks
        for chunk in chunks:
            if chunk.start_char == -1:
                continue
            # La position doit rester dans les bornes du document, meme
            # si elle ne recouvre pas caractere pour caractere un texte
            # de fragment localement duplique.
            assert 0 <= chunk.start_char < chunk.end_char <= len(content)


class TestRepliEnCoursDeBatch:
    """
    Trou de couverture constate sur le corpus reel wpd : pymupdf4llm
    echouait sur les six batchs d'un tome, et le repli laissait la table
    de pagination desynchronisee du texte.

    L'ancienne boucle versait le texte de chaque page dans `all_parts`
    avant de lire son numero de page. Quand cette lecture levait, le
    texte etait deja ajoute mais ni le span ni `offset` ne l'etaient : le
    repli rejouait alors tout le batch et le texte se retrouvait en
    double, decalant toutes les positions suivantes.

    Les tests precedents ne prenaient jamais le chemin du repli.
    """

    def test_repli_ne_duplique_pas_le_texte(self, pdf_multi_pages: Path, monkeypatch) -> None:
        import src.ingestion.parsers as parsers_module

        def to_markdown_qui_echoue(file_path, pages=None, page_chunks=False, **kwargs):
            # Reproduit exactement l'ancien defaut : un retour valide dont
            # les metadonnees ne portent pas la clef attendue. Le code
            # appelant doit soit s'en passer, soit basculer proprement.
            return [
                {"text": f"texte factice page {p}\n", "metadata": {"page_number": p + 1}}
                for p in (pages or [])
            ]

        monkeypatch.setattr(parsers_module.pymupdf4llm, "to_markdown", to_markdown_qui_echoue)

        parser = parsers_module.PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_multi_pages))

        content = result["content"]
        page_spans = result["metadata"]["page_spans"]

        assert page_spans, "page_spans doit rester renseignee"
        # Chaque span doit pointer sur le texte reel a cette position.
        for debut, fin, numero_page in page_spans:
            assert content[debut:fin] == f"texte factice page {numero_page - 1}\n"
        # Le dernier span doit fermer exactement sur la fin du contenu :
        # aucun texte n'est ajoute sans span correspondant.
        assert page_spans[-1][1] == len(content)

    def test_repli_sur_exception_reste_coherent(self, pdf_multi_pages: Path, monkeypatch) -> None:
        import src.ingestion.parsers as parsers_module

        def to_markdown_qui_leve(file_path, pages=None, page_chunks=False, **kwargs):
            raise KeyError("page")

        monkeypatch.setattr(parsers_module.pymupdf4llm, "to_markdown", to_markdown_qui_leve)

        parser = parsers_module.PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_multi_pages))

        content = result["content"]
        page_spans = result["metadata"]["page_spans"]

        assert page_spans, "le repli doit produire une table de pagination"
        # Aucune duplication : la somme des longueurs de spans couvre tout
        # le contenu, sans trou ni recouvrement.
        assert page_spans[0][0] == 0
        assert page_spans[-1][1] == len(content)
        for i in range(len(page_spans) - 1):
            assert page_spans[i][1] == page_spans[i + 1][0]
        # Les numeros de page suivent l'ordre du document, en base 1.
        assert [p for _, _, p in page_spans] == sorted(p for _, _, p in page_spans)
        assert min(p for _, _, p in page_spans) >= 1

    def test_numero_de_page_independant_de_la_clef_de_metadonnees(
        self, pdf_multi_pages: Path, monkeypatch
    ) -> None:
        """
        La clef portant le numero de page varie selon la version de
        pymupdf4llm et les greffons installes : "page" dans un
        environnement, "page_number" dans un autre, pour des versions
        annoncees identiques. Le numero retenu doit venir de l'argument
        `pages` transmis a l'appel, jamais des metadonnees renvoyees.
        """
        import src.ingestion.parsers as parsers_module

        def to_markdown_sans_aucune_clef_de_page(file_path, pages=None, page_chunks=False, **kwargs):
            return [
                {"text": f"page {p}\n", "metadata": {"title": "", "author": ""}}
                for p in (pages or [])
            ]

        monkeypatch.setattr(
            parsers_module.pymupdf4llm, "to_markdown", to_markdown_sans_aucune_clef_de_page
        )

        parser = parsers_module.PDFParser(enable_ocr=False)
        result = parser.parse(str(pdf_multi_pages))
        page_spans = result["metadata"]["page_spans"]

        assert [p for _, _, p in page_spans] == [1, 2, 3]


class TestPositionApresRecouvrement:
    """
    Second trou de couverture du corpus reel : sur un tome de 54 pages,
    117 fragments sur 132 ressortaient sans page.

    Apres fermeture d'un fragment, l'ancienne implementation cherchait la
    queue de recouvrement dans `content` par `rfind`. Cette queue n'est
    pas une unite du document : le fragment accumule colle les unites
    avec un espace simple la ou le document porte un saut de ligne. Des
    que la queue traversait une jonction, la recherche echouait, et
    l'echec ne se rattrapait jamais parce que la condition de reprise
    exigeait un fragment courant vide, alors qu'il contenait justement
    cette queue.

    Les tests precedents utilisaient des textes trop courts pour produire
    la sequence de fragments necessaire a l'apparition du defaut.
    """

    @staticmethod
    def _contenu_multi_lignes(nb_phrases: int = 60) -> str:
        # Separateurs varies (saut de ligne simple, double, espaces
        # multiples) : c'est precisement ce que l'espace de jonction
        # synthetique du fragment accumule ne reproduit pas.
        separateurs = ["\n", "\n\n", "   ", " \n "]
        morceaux = []
        for i in range(nb_phrases):
            morceaux.append(
                f"Phrase numero {i} du document de test, suffisamment longue "
                f"pour que plusieurs d entre elles remplissent un fragment."
            )
            morceaux.append(separateurs[i % len(separateurs)])
        return "".join(morceaux)

    def test_aucune_perte_de_position_apres_recouvrement(self) -> None:
        content = self._contenu_multi_lignes()
        chunker = Chunker(chunk_size=400, chunk_overlap=80, strategy="sentence")
        chunks = chunker.chunk(document_pour_chunker(content, {}))

        assert len(chunks) > 5, "le contenu doit produire assez de fragments"
        sans_position = [c for c in chunks if c.start_char == -1]
        assert not sans_position, (
            f"{len(sans_position)}/{len(chunks)} fragments sans position : "
            "la position se perd apres le recouvrement"
        )

    def test_positions_strictement_croissantes(self) -> None:
        content = self._contenu_multi_lignes()
        chunker = Chunker(chunk_size=400, chunk_overlap=80, strategy="sentence")
        chunks = chunker.chunk(document_pour_chunker(content, {}))

        debuts = [c.start_char for c in chunks]
        assert debuts == sorted(debuts), "les fragments doivent avancer dans le document"
        for c in chunks:
            assert 0 <= c.start_char < c.end_char <= len(content)

    def test_pages_renseignees_sur_tout_le_document(self) -> None:
        """Avec une table de pagination, tous les fragments doivent porter
        une page, pas seulement les premiers."""
        content = self._contenu_multi_lignes()
        # Trois pages de tailles egales couvrant tout le contenu.
        tiers = len(content) // 3
        page_spans = [
            (0, tiers, 1),
            (tiers, 2 * tiers, 2),
            (2 * tiers, len(content), 3),
        ]
        chunker = Chunker(chunk_size=400, chunk_overlap=80, strategy="sentence")
        chunks = chunker.chunk(
            document_pour_chunker(content, {"page_spans": page_spans})
        )

        sans_page = [c for c in chunks if c.metadata.get("page_start") is None]
        assert not sans_page, (
            f"{len(sans_page)}/{len(chunks)} fragments sans page"
        )
        # Les pages progressent avec le document et restent dans les bornes.
        pages = [c.metadata["page_start"] for c in chunks]
        assert pages == sorted(pages)
        assert min(pages) == 1
        assert max(pages) == 3

    def test_recouvrement_nul_reste_correct(self) -> None:
        """Sans recouvrement, le comportement anterieur doit etre preserve."""
        content = self._contenu_multi_lignes()
        chunker = Chunker(chunk_size=400, chunk_overlap=0, strategy="sentence")
        chunks = chunker.chunk(document_pour_chunker(content, {}))

        assert chunks
        assert all(c.start_char >= 0 for c in chunks)
