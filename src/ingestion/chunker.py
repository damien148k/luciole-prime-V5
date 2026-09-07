"""
Document Chunker — Intelligent text chunking strategies
V3 : Chunking adaptatif par format de fichier

Enrichit chaque chunk avec le contexte du fichier (chemin + nom) pour améliorer la recherche.
Stratégies spéciales :
  - XLSX : l'en-tête (ligne 1) est répétée dans chaque chunk
  - PPTX : 1 chunk = 1 slide complet (titre + corps + notes)
  - MSG/EML : métadonnées (De, À, Objet, Date) incluses dans chaque chunk
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
import bisect
import re
import yaml


# Les analyseurs de documents annoncent un type qui ne correspond pas toujours
# a la clef utilisee dans chunking_strategies de settings.yaml. Sans cette
# table, les strategies 'xlsx' et 'txt' n'etaient jamais appliquees.
TYPE_ALIASES = {
    "excel": "xlsx",
    "text": "txt",
    "doc": "docx",
    "ppt": "pptx",
    "email": "msg",
}


@dataclass
class Chunk:
    """Represents a document chunk."""
    text: str
    text_with_context: str
    chunk_id: str
    document_id: str
    file_path: str
    file_name: str
    start_char: int
    end_char: int
    metadata: Dict


class Chunker:
    """
    Intelligent document chunker with multiple strategies.
    V3 : supporte le chunking adaptatif par type de fichier via settings.yaml.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "sentence",
        include_file_context: bool = True,
        adaptive: bool = False,
        chunking_strategies: Dict = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.include_file_context = include_file_context
        self.adaptive = adaptive
        self.chunking_strategies = chunking_strategies or {}

        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

        logger.info(
            f"Chunker initialized: size={chunk_size}, overlap={chunk_overlap}, "
            f"strategy={strategy}, context={include_file_context}, adaptive={adaptive}"
        )
        if adaptive:
            if self.chunking_strategies:
                detail = ", ".join(
                    f"{t}={c.get('chunk_size', chunk_size)}/{c.get('chunk_overlap', chunk_overlap)}"
                    f":{c.get('strategy', strategy)}"
                    for t, c in sorted(self.chunking_strategies.items())
                )
                logger.info(f"Chunker adaptatif : {detail}")
            else:
                logger.warning(
                    "Chunker adaptatif demande mais aucune strategie par type "
                    "n'a ete fournie : les valeurs globales seront appliquees "
                    "a tous les formats."
                )

    @classmethod
    def from_config(cls, config_path: str = "config/settings.yaml") -> "Chunker":
        """Construit un Chunker depuis le fichier de configuration complet."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        chunking = config.get("chunking", {})
        return cls(
            chunk_size=chunking.get("chunk_size", 512),
            chunk_overlap=chunking.get("chunk_overlap", 50),
            strategy=chunking.get("strategy", "sentence"),
            include_file_context=chunking.get("include_file_context", True),
            adaptive=chunking.get("adaptive", False),
            chunking_strategies=config.get("chunking_strategies", {}),
        )

    def _resolve_strategy(self, doc_type: str) -> Dict:
        """Résout la stratégie de chunking pour un type de document donné."""
        if not self.adaptive or not self.chunking_strategies:
            return {
                "strategy": self.strategy,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

        type_lower = TYPE_ALIASES.get(doc_type.lower(), doc_type.lower())
        if type_lower in self.chunking_strategies:
            cfg = self.chunking_strategies[type_lower]
            return {
                "strategy": cfg.get("strategy", self.strategy),
                "chunk_size": cfg.get("chunk_size", self.chunk_size),
                "chunk_overlap": cfg.get("chunk_overlap", self.chunk_overlap),
                "rows_per_chunk": cfg.get("rows_per_chunk"),
                "include_header": cfg.get("include_header", True),
            }

        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def chunk(self, document: Dict) -> List[Chunk]:
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        file_path = metadata.get("file_path", "")
        file_name = metadata.get("file_name", "unknown")
        doc_id = file_name
        doc_type = metadata.get("type", "")

        if not content.strip():
            logger.warning(f"Empty content for document: {doc_id}")
            return []

        file_context = self._build_file_context(file_path, file_name, metadata)

        resolved = self._resolve_strategy(doc_type)
        strat = resolved["strategy"]

        if strat == "slide":
            return self._chunk_by_slide(content, doc_id, file_path, file_name, file_context, metadata, resolved)
        elif strat == "email":
            return self._chunk_email(content, doc_id, file_path, file_name, file_context, metadata, resolved)
        elif strat == "paragraph":
            return self._chunk_by_paragraph(content, doc_id, file_path, file_name, file_context, metadata, resolved)
        elif strat == "sentence":
            return self._chunk_by_sentence(content, doc_id, file_path, file_name, file_context, metadata, resolved)
        else:
            return self._chunk_fixed(content, doc_id, file_path, file_name, file_context, metadata, resolved)

    # =========================================================================
    # FILE CONTEXT
    # =========================================================================

    def _build_file_context(self, file_path: str, file_name: str, metadata: Dict) -> str:
        path_obj = Path(file_path) if file_path else None
        parent_path = ""
        if path_obj and path_obj.parent:
            parts = path_obj.parent.parts[-3:]
            parent_path = "/".join(parts) if parts else ""

        doc_type = metadata.get("type", "document").upper()

        context_parts = [f"Fichier: {file_name}"]
        if parent_path:
            context_parts.append(f"Chemin: {parent_path}")
        context_parts.append(f"Type: {doc_type}")

        if metadata.get("title"):
            context_parts.append(f"Titre: {metadata['title']}")
        if metadata.get("author"):
            context_parts.append(f"Auteur: {metadata['author']}")
        if metadata.get("subject"):
            context_parts.append(f"Sujet: {metadata['subject']}")

        return "[" + " | ".join(context_parts) + "]"

    @staticmethod
    def _pages_couvertes(page_spans: Optional[List[Tuple[int, int, int]]], start_char: int, end_char: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Determine les numeros de page couvrant l'intervalle [start_char, end_char).

        `page_spans` est une liste ordonnee de tuples (debut, fin, page),
        triee par `debut` croissant (construite ainsi par les parsers). On
        utilise `bisect` sur la liste des bornes de debut pour localiser en
        O(log n) les spans qui recouvrent l'intervalle, plutot qu'un
        balayage lineaire.

        Retourne (page_start, page_end), chacun None si aucune page n'a pu
        etre etablie (page_spans absente/vide, ou position -1 signifiant
        une position non retrouvee par le chunker).
        """
        if not page_spans or start_char < 0 or end_char < 0 or end_char <= start_char:
            return None, None

        debuts = [span[0] for span in page_spans]

        # Page couvrant le premier caractere du fragment : dernier span dont
        # le debut est <= start_char.
        idx_debut = bisect.bisect_right(debuts, start_char) - 1
        # Page couvrant le dernier caractere du fragment (end_char exclu) :
        # dernier span dont le debut est <= end_char - 1.
        idx_fin = bisect.bisect_right(debuts, end_char - 1) - 1

        if idx_debut < 0 or idx_fin < 0:
            return None, None

        page_start = page_spans[idx_debut][2]
        page_end = page_spans[idx_fin][2]
        return page_start, page_end

    @staticmethod
    def _reprendre_apres_recouvrement(
        unites_du_fragment: List[Tuple[int, int, int]],
        longueur_recouvrement: int,
    ) -> Tuple[Optional[int], List[Tuple[int, int, int]]]:
        """
        Position reelle du debut de la queue de recouvrement, et etat des
        unites a reporter dans le fragment suivant.

        La queue est un suffixe de `current_chunk`, chaine construite en
        collant les unites avec un espace simple. Ce separateur synthetique
        ne correspond pas au separateur reel du document (saut de ligne,
        blancs multiples), ce qui rend la queue introuvable par
        `content.find` des qu'elle traverse une jonction. Sa position est
        donc deduite des unites qui la composent, chacune connaissant sa
        position reelle et sa contribution exacte a `current_chunk`.

        On remonte les unites depuis la fin en cumulant leurs contributions
        jusqu'a couvrir la longueur de la queue. L'unite qui la fait basculer
        n'y entre generalement qu'en partie : le decalage `d` a l'interieur
        de sa contribution donne directement la position reelle recherchee,
        `position + d`, tant que `d` tombe dans le texte de l'unite. Si `d`
        tombe sur l'espace de jonction qui la suit, la queue commence en
        realite a l'unite suivante.

        Retourne (position ou None, unites reportees). La position vaut None
        quand la queue est vide ou quand les unites disponibles ne suffisent
        pas a la couvrir : le fragment suivant prendra alors la position de
        sa premiere unite entiere, sans jamais rester bloque sans position.
        """
        if longueur_recouvrement <= 0 or not unites_du_fragment:
            return None, []

        cumul = 0
        indice = len(unites_du_fragment)
        for i in range(len(unites_du_fragment) - 1, -1, -1):
            cumul += unites_du_fragment[i][1]
            indice = i
            if cumul >= longueur_recouvrement:
                break
        else:
            # Les unites connues ne couvrent pas toute la queue : cela
            # signifie que le fragment ferme contenait deja un report dont
            # la chaine de positions etait rompue.
            return None, []

        position, contribution, longueur_texte = unites_du_fragment[indice]
        decalage = cumul - longueur_recouvrement

        if decalage <= longueur_texte:
            reportees = [
                (position + decalage, contribution - decalage, longueur_texte - decalage)
            ] + list(unites_du_fragment[indice + 1:])
            return position + decalage, reportees

        # Le decalage tombe sur l'espace de jonction suivant l'unite : la
        # queue commence a l'unite d'apres, si elle existe.
        reportees = list(unites_du_fragment[indice + 1:])
        if not reportees:
            return None, []
        return reportees[0][0], reportees

    def _make_chunk(self, text: str, file_context: str, doc_id: str, file_path: str, file_name: str, metadata: Dict, chunk_idx: int, start_char: int, end_char: int) -> Chunk:
        text_with_context = f"{file_context}\n{text}" if self.include_file_context else text

        # page_spans decrit la pagination du document entier (table), pas du
        # fragment : on la lit ici pour calculer page_start/page_end puis on
        # la retire explicitement des metadonnees du fragment pour ne pas la
        # dupliquer dans chaque chunk.
        page_spans = metadata.get("page_spans")
        page_start, page_end = self._pages_couvertes(page_spans, start_char, end_char)

        chunk_metadata = {k: v for k, v in metadata.items() if k != "page_spans"}
        chunk_metadata["chunk_index"] = chunk_idx
        chunk_metadata["page_start"] = page_start
        chunk_metadata["page_end"] = page_end

        return Chunk(
            text=text,
            text_with_context=text_with_context,
            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
            document_id=doc_id,
            file_path=file_path,
            file_name=file_name,
            start_char=start_char,
            end_char=end_char,
            metadata=chunk_metadata
        )

    # =========================================================================
    # SENTENCE CHUNKING
    # =========================================================================

    def _split_oversized(self, text: str, limit: int, overlap: int) -> List[str]:
        """
        Decoupe une unite (phrase, paragraphe) plus longue que la limite.

        Le decoupage en phrases repose sur la ponctuation. Un tableau, un
        sommaire, une legende de carte ou une liste a puces n'en contiennent
        pas : l'unite produite peut alors faire plusieurs dizaines de milliers
        de caracteres. Sans borne, ce fragment part tel quel vers l'encodeur
        et sature la memoire du GPU.

        La coupure est cherchee en fin de fenetre d'abord sur un saut de
        ligne (frontiere de ligne de tableau Markdown : une ligne coupee en
        deux produit deux demi-lignes inexploitables), sinon sur un espace,
        afin de ne pas tronquer un mot au milieu.
        """
        if limit <= 0 or len(text) <= limit:
            return [text]

        pas = max(1, limit - max(0, overlap))
        morceaux: List[str] = []
        debut = 0

        while debut < len(text):
            fin = min(debut + limit, len(text))
            if fin < len(text):
                fenetre = text.rfind("\n", debut + pas // 2, fin)
                if fenetre <= debut:
                    fenetre = text.rfind(" ", debut + pas // 2, fin)
                if fenetre > debut:
                    fin = fenetre
            morceau = text[debut:fin].strip()
            if morceau:
                morceaux.append(morceau)
            if fin >= len(text):
                break
            suivant = max(fin - max(0, overlap), debut + 1)
            if overlap > 0 and suivant < len(text) and not text[suivant - 1].isspace():
                blanc = text.find(" ", suivant, fin)
                if blanc != -1 and blanc + 1 < fin:
                    suivant = blanc + 1
            debut = suivant

        return morceaux or [text[:limit]]

    @staticmethod
    def _queue_recouvrement(texte: str, co: int) -> str:
        """
        Queue de recouvrement d'un fragment, alignee sur un debut de mot.

        Le recouvrement reprend les `co` derniers caracteres du fragment
        precedent. Pris tels quels, ils tombent presque toujours au milieu
        d'un mot : le fragment suivant commencait par "ces et reseau" pour
        "acces et reseau", ou "vant d'etre delivree" pour "avant d'etre
        delivree". Constate sur les etudes d'impact du corpus wpd, ou six
        extraits sur quinze transmis au modele demarraient ainsi.

        Le cout est double : l'extrait presente au modele s'ouvre sur un
        mot mutile, et le vecteur dense est calcule sur ce meme texte.

        La queue est donc avancee jusqu'a la premiere frontiere de mot.
        Elle en ressort au plus egale a `co`, jamais plus longue, ce qui
        preserve la borne de taille des fragments. Si aucune frontiere de
        mot ne s'y trouve, le recouvrement est abandonne plutot que
        tronque.
        """
        if co <= 0 or not texte:
            return ""
        if len(texte) <= co:
            # Comportement anterieur conserve : un fragment plus court que
            # le recouvrement ne produit pas de queue. Le renvoyer entier
            # rendrait le fragment suivant identique au precedent.
            return ""

        debut = len(texte) - co
        # Deja aligne : le caractere precedent est un blanc.
        if texte[debut - 1].isspace():
            return texte[debut:]

        avance = re.search(r"\s", texte[debut:])
        if avance is None:
            # Aucune frontiere de mot dans la queue : elle tombe a
            # l'interieur d'un jeton unique plus long qu'elle (URL,
            # identifiant, cellule de tableau compacte), ou le recouvrement
            # demande est plus court qu'un mot. Recouvrir avec une bribe de
            # jeton n'apporte aucun contexte au fragment suivant et lui fait
            # ouvrir sur un mot mutile. On renonce au recouvrement.
            return ""

        return texte[debut + avance.end():]

    def _localiser_unites(self, content: str, unites: List[str]) -> List[Tuple[str, int]]:
        """
        Retrouve la position reelle de chaque unite dans `content`.

        Les unites (phrases ou paragraphes) sont des sous-chaines de
        `content` : le motif de decoupe ne consomme que des blancs et
        `_split_oversized` decoupe et strip une sous-chaine. Une recherche
        incrementale `content.find(unite, curseur)` retrouve donc la
        position reelle de chaque unite en avancant le curseur.

        `_split_oversized` produit des morceaux qui se chevauchent dans
        `content` (le recouvrement demande fait qu'un morceau commence
        avant la fin du precedent) : le curseur n'avance donc qu'au debut
        de l'unite trouvee (+1), jamais jusqu'a sa fin, pour ne pas rendre
        introuvable un morceau suivant chevauchant. Les debuts de morceaux
        successifs sont toujours strictement croissants par construction
        de `_split_oversized`, donc cette avance minimale suffit a garantir
        la progression sans confondre deux occurrences identiques.

        Si la recherche echoue (retour -1), la position vaut -1 : les
        fragments concernes seront alors marques sans page plutot que de
        recevoir une position devinee.

        Retourne une liste de tuples (unite, position) dans le meme ordre
        que `unites`.
        """
        localisees = []
        curseur = 0
        for unite in unites:
            pos = content.find(unite, curseur)
            if pos != -1:
                curseur = pos + 1
            localisees.append((unite, pos))
        return localisees

    def _chunk_by_sentence(self, content, doc_id, file_path, file_name, file_context, metadata, resolved):
        cs = resolved.get("chunk_size", self.chunk_size)
        co = resolved.get("chunk_overlap", self.chunk_overlap)

        sentences = self.sentence_pattern.split(content)
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        # Position reelle du debut du chunk courant dans `content`. None si
        # elle n'a pas pu etre etablie (unite introuvable) : le chunk sera
        # alors marque sans position plutot que de deviner.
        chunk_start_reel: Optional[int] = None
        # Fin reelle du contenu deja accumule dans current_chunk (avant
        # d'ajouter l'espace de separation final).
        fin_reelle: Optional[int] = None
        # Unites presentes dans current_chunk, sous la forme
        # (position_reelle, contribution_a_current_chunk, longueur_du_texte).
        # La contribution compte l'espace de jonction ajoute apres l'unite ;
        # la longueur du texte ne le compte pas. L'ecart entre les deux est
        # ce qui rend la queue de recouvrement introuvable par recherche
        # textuelle et impose de calculer sa position.
        unites_du_fragment: List[Tuple[int, int, int]] = []

        unites = []
        surdimensionnees = 0
        for brute in sentences:
            brute = brute.strip()
            if not brute:
                continue
            if len(brute) > cs:
                surdimensionnees += 1
                # Le fragment suivant demarre avec le recouvrement du
                # precedent : on lui reserve sa place pour que la somme des
                # deux ne depasse jamais la taille demandee.
                unites.extend(self._split_oversized(brute, max(1, cs - co), co))
            else:
                unites.append(brute)

        if surdimensionnees:
            logger.warning(
                f"{doc_id} : {surdimensionnees} unite(s) sans ponctuation depassant "
                f"{cs} caracteres ont ete redecoupees (tableaux, sommaires ou listes)."
            )

        localisees = self._localiser_unites(content, unites)

        for sentence, pos_reelle in localisees:
            if len(current_chunk) + len(sentence) > cs and current_chunk:
                text = current_chunk.strip()
                chunks.append(self._make_chunk(
                    text, file_context, doc_id, file_path, file_name, metadata, chunk_idx,
                    chunk_start_reel if chunk_start_reel is not None else -1,
                    fin_reelle if fin_reelle is not None else -1,
                ))
                chunk_idx += 1
                # La queue de recouvrement n'est pas une unite du document :
                # `current_chunk` colle les unites avec un espace simple la
                # ou le document porte un saut de ligne ou plusieurs blancs.
                # La rechercher dans `content` echouait des qu'elle traversait
                # une jonction, et l'echec ne se rattrapait jamais, la
                # condition de reprise portant sur un `current_chunk` vide
                # alors qu'il contient precisement cette queue. Sur les
                # etudes d'impact du corpus wpd, 117 fragments sur 132
                # ressortaient ainsi sans page.
                #
                # Sa position est desormais calculee, pas cherchee : les
                # unites du fragment ferme sont connues avec leur position
                # reelle et leur contribution exacte a `current_chunk`.
                overlap_text = self._queue_recouvrement(current_chunk, co)
                chunk_start_reel, unites_du_fragment = self._reprendre_apres_recouvrement(
                    unites_du_fragment, len(overlap_text)
                )
                current_chunk = overlap_text

            if chunk_start_reel is None and pos_reelle != -1:
                chunk_start_reel = pos_reelle

            current_chunk += sentence + " "
            if pos_reelle != -1:
                fin_reelle = pos_reelle + len(sentence)
                unites_du_fragment.append((pos_reelle, len(sentence) + 1, len(sentence)))
            else:
                # Unite non localisee : la chaine de positions du fragment
                # est rompue, on repart de la prochaine unite localisee.
                unites_du_fragment = []

        if current_chunk.strip():
            text = current_chunk.strip()
            chunks.append(self._make_chunk(
                text, file_context, doc_id, file_path, file_name, metadata, chunk_idx,
                chunk_start_reel if chunk_start_reel is not None else -1,
                fin_reelle if fin_reelle is not None else -1,
            ))

        chunks = self._reinjecter_entetes_tableaux(content, chunks, file_context)

        logger.info(f"Created {len(chunks)} chunks for document: {doc_id} (sentence)")
        return chunks

    # =========================================================================
    # TABLEAUX MARKDOWN
    # =========================================================================

    # Une ligne de tableau Markdown commence par un pipe (apres espaces
    # eventuels). pymupdf4llm produit ce format pour les tableaux PDF.
    _LIGNE_TABLEAU_RE = re.compile(r"^\s*\|")
    # Ligne de separation d'en-tete : |---|---| (tirets, deux-points, espaces)
    _SEPARATEUR_TABLEAU_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")

    @classmethod
    def _detecter_tableaux(cls, content: str) -> List[Dict]:
        """
        Repere les blocs de tableaux Markdown dans `content`.

        Retourne une liste de dicts {debut, fin, entete} ou `debut`/`fin`
        sont les offsets du bloc dans `content` et `entete` le texte des
        lignes d'en-tete (ligne de titres + ligne de separation
        |---|---| si presente), destine a etre re-injecte dans les
        fragments qui commencent au milieu du tableau.
        """
        tableaux: List[Dict] = []
        debut_bloc: Optional[int] = None
        pos = 0
        for ligne in content.split("\n"):
            if cls._LIGNE_TABLEAU_RE.match(ligne):
                if debut_bloc is None:
                    debut_bloc = pos
            else:
                if debut_bloc is not None:
                    tableaux.append({"debut": debut_bloc, "fin": pos})
                    debut_bloc = None
            pos += len(ligne) + 1
        if debut_bloc is not None:
            tableaux.append({"debut": debut_bloc, "fin": len(content)})

        for tableau in tableaux:
            lignes = content[tableau["debut"]:tableau["fin"]].split("\n")
            entete_lignes = lignes[:1]
            if len(lignes) > 1 and cls._SEPARATEUR_TABLEAU_RE.match(lignes[1]):
                entete_lignes.append(lignes[1])
            tableau["entete"] = "\n".join(entete_lignes)
            # Offset de fin de l'en-tete dans content : un fragment qui
            # commence apres cet offset ne contient pas l'en-tete.
            tableau["fin_entete"] = tableau["debut"] + len(tableau["entete"])

        return tableaux

    def _reinjecter_entetes_tableaux(self, content: str, chunks: List[Chunk], file_context: str) -> List[Chunk]:
        """
        Re-injecte l'en-tete d'un tableau Markdown dans les fragments qui
        commencent au milieu de ce tableau.

        Un long tableau (inventaire des monuments historiques, matrice de
        sensibilite) est decoupe en plusieurs fragments : sans rappel des
        colonnes, les fragments suivants sont des lignes de cellules
        orphelines, difficiles a interpreter pour l'encodeur comme pour le
        LLM. L'en-tete est prefixe au texte du fragment, a la maniere du
        contexte fichier deja prefixe par _make_chunk : c'est un contexte
        de presentation, les offsets start_char/end_char continuent donc de
        decrire la zone reelle du document (pagination inchangee).
        """
        if not chunks:
            return chunks

        tableaux = self._detecter_tableaux(content)
        if not tableaux:
            return chunks

        nb_reinjections = 0
        for chunk in chunks:
            if chunk.start_char is None or chunk.start_char < 0:
                continue
            for tableau in tableaux:
                if tableau["debut"] <= chunk.start_char < tableau["fin"]:
                    if chunk.start_char > tableau["fin_entete"]:
                        chunk.text = f"{tableau['entete']}\n{chunk.text}"
                        chunk.text_with_context = (
                            f"{file_context}\n{chunk.text}"
                            if self.include_file_context else chunk.text
                        )
                        nb_reinjections += 1
                    break

        if nb_reinjections:
            logger.info(
                f"{nb_reinjections} fragment(s) ont recu l'en-tete de leur tableau"
            )
        return chunks

    # =========================================================================
    # PARAGRAPH CHUNKING
    # =========================================================================

    def _chunk_by_paragraph(self, content, doc_id, file_path, file_name, file_context, metadata, resolved):
        cs = resolved.get("chunk_size", self.chunk_size)
        co = resolved.get("chunk_overlap", self.chunk_overlap)

        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        # Position reelle du debut du chunk courant dans `content`, et fin
        # reelle du contenu deja accumule (avant le "\n\n" de jonction).
        # None si non etablie : le chunk sera marque sans page plutot que
        # de deviner une position.
        chunk_start_reel: Optional[int] = None
        fin_reelle: Optional[int] = None

        unites = []
        surdimensionnes = 0
        for brut in paragraphs:
            brut = brut.strip()
            if not brut:
                continue
            if len(brut) > cs:
                surdimensionnes += 1
                unites.extend(self._split_oversized(brut, cs, co))
            else:
                unites.append(brut)

        if surdimensionnes:
            logger.warning(
                f"{doc_id} : {surdimensionnes} paragraphe(s) depassant {cs} "
                f"caracteres ont ete redecoupes."
            )

        localisees = self._localiser_unites(content, unites)

        for para, pos_reelle in localisees:
            if len(current_chunk) + len(para) > cs and current_chunk:
                text = current_chunk.strip()
                chunks.append(self._make_chunk(
                    text, file_context, doc_id, file_path, file_name, metadata, chunk_idx,
                    chunk_start_reel if chunk_start_reel is not None else -1,
                    fin_reelle if fin_reelle is not None else -1,
                ))
                chunk_idx += 1
                chunk_start_reel = None
                fin_reelle = None
                current_chunk = ""

            if chunk_start_reel is None and pos_reelle != -1:
                chunk_start_reel = pos_reelle

            current_chunk += para + "\n\n"
            if pos_reelle != -1:
                fin_reelle = pos_reelle + len(para)

        if current_chunk.strip():
            text = current_chunk.strip()
            chunks.append(self._make_chunk(
                text, file_context, doc_id, file_path, file_name, metadata, chunk_idx,
                chunk_start_reel if chunk_start_reel is not None else -1,
                fin_reelle if fin_reelle is not None else -1,
            ))

        logger.info(f"Created {len(chunks)} chunks for document: {doc_id} (paragraph)")
        return chunks

    # =========================================================================
    # SLIDE CHUNKING (PPTX)
    # =========================================================================

    def _chunk_by_slide(self, content, doc_id, file_path, file_name, file_context, metadata, resolved):
        slide_pattern = re.compile(r'---\s*Slide\s+\d+\s*---')
        parts = slide_pattern.split(content)
        headers = slide_pattern.findall(content)

        chunks = []
        chunk_idx = 0
        pos = 0

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                pos += len(part) + (len(headers[i]) if i < len(headers) else 0)
                continue

            slide_header = headers[i - 1] if i > 0 and i - 1 < len(headers) else ""
            text = f"{slide_header}\n{part}".strip() if slide_header else part

            chunks.append(self._make_chunk(text, file_context, doc_id, file_path, file_name, metadata, chunk_idx, pos, pos + len(text)))
            chunk_idx += 1
            pos += len(part) + (len(headers[i]) if i < len(headers) else 0)

        if not chunks and content.strip():
            chunks.append(self._make_chunk(content.strip(), file_context, doc_id, file_path, file_name, metadata, 0, 0, len(content)))

        logger.info(f"Created {len(chunks)} chunks for document: {doc_id} (slide)")
        return chunks

    # =========================================================================
    # EMAIL CHUNKING (MSG / EML)
    # =========================================================================

    def _chunk_email(self, content, doc_id, file_path, file_name, file_context, metadata, resolved):
        cs = resolved.get("chunk_size", 800)

        lines = content.split("\n")
        meta_lines = []
        body_start = 0

        for i, line in enumerate(lines):
            if line.strip().lower().startswith(("subject:", "from:", "to:", "date:", "de:", "à:", "objet:")):
                meta_lines.append(line.strip())
                body_start = i + 1
            elif not line.strip() and meta_lines:
                body_start = i + 1
                break

        email_header = "\n".join(meta_lines)
        body = "\n".join(lines[body_start:]).strip()

        if not body:
            text = email_header if email_header else content.strip()
            return [self._make_chunk(text, file_context, doc_id, file_path, file_name, metadata, 0, 0, len(content))]

        chunks = []
        chunk_idx = 0
        sentences = self.sentence_pattern.split(body)
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) > cs and current:
                full_text = f"{email_header}\n\n{current.strip()}" if email_header else current.strip()
                chunks.append(self._make_chunk(full_text, file_context, doc_id, file_path, file_name, metadata, chunk_idx, 0, len(full_text)))
                chunk_idx += 1
                current = ""

            current += sentence + " "

        if current.strip():
            full_text = f"{email_header}\n\n{current.strip()}" if email_header else current.strip()
            chunks.append(self._make_chunk(full_text, file_context, doc_id, file_path, file_name, metadata, chunk_idx, 0, len(full_text)))

        logger.info(f"Created {len(chunks)} chunks for document: {doc_id} (email)")
        return chunks

    # =========================================================================
    # FIXED CHUNKING
    # =========================================================================

    def _chunk_fixed(self, content, doc_id, file_path, file_name, file_context, metadata, resolved):
        cs = resolved.get("chunk_size", self.chunk_size)
        co = resolved.get("chunk_overlap", self.chunk_overlap)

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = min(start + cs, len(content))
            text = content[start:end].strip()
            chunks.append(self._make_chunk(text, file_context, doc_id, file_path, file_name, metadata, chunk_idx, start, end))
            chunk_idx += 1
            start = end - co
            if start >= len(content) - co:
                break

        logger.info(f"Created {len(chunks)} chunks for document: {doc_id} (fixed)")
        return chunks
