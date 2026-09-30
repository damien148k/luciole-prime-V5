"""Generateur LLM (Ollama natif ou OpenAI-compatible), configure par Settings.

Repris de V3 (`generation/llm.py`) : memes deux chemins d'appel, meme
prompt RAG lu dans prompts.yaml. Difference V4 : les reglages sont lus
sur l'objet Settings recu, jamais sur une relecture privee du YAML.

v3.1 :
- `last_usage` : mesure de la derniere generation (tokens d'entree et de
  sortie, `done_reason`, `tronque`) lisible par query2 apres chaque
  `analyze()` ; c'est la seule facon de savoir qu'une reponse a ete
  arretee par max_tokens.
- `surcharge(**options)` : contexte de surcharge temporaire des options
  d'echantillonnage (relance d'un bloc qui boucle sous repeat_penalty).
- `llm.options` (settings.yaml) : options Ollama supplementaires envoyees
  a chaque appel (repeat_penalty, repeat_last_n, top_k, ...).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from src.config.prompts import load_prompts
from src.config.settings import Settings

# Options Ollama transposables sur l'API OpenAI-compatible.
_VERS_OPENAI = {"frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty", "top_p": "top_p"}


class LLMGenerator:
    def __init__(self, settings: Settings, prompts=None):
        llm = settings.section("llm")
        base = llm.get("base_url", "http://ollama:11434").rstrip("/")
        self.base_url = base if base.endswith("/v1") else f"{base}/v1"
        self.native_base = base[:-3] if base.endswith("/v1") else base
        self.api_format = llm.get("api_format", "openai")
        self.model = llm.get("model", "")
        self.temperature = llm.get("temperature", 0.0)
        self.max_tokens = llm.get("max_tokens", 4096)
        self.timeout = llm.get("timeout", 300)
        self.num_ctx = llm.get("num_ctx", 32768)
        self.seed = llm.get("seed", 42)
        self.options_base: Dict[str, Any] = dict(llm.get("options") or {})
        self._surcharge: Dict[str, Any] = {}
        self.last_usage: Dict[str, Any] = {}
        self.prompts = prompts if prompts is not None else self._charger_prompts()
        logger.info(f"LLMGenerator : model={self.model}, base={base}, format={self.api_format}, "
                    f"options={self.options_base}")

    @staticmethod
    def _charger_prompts():
        try:
            return load_prompts()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"prompts.yaml illisible ({e}) : gabarits integres")
            return None

    # ------------------------------------------------------------------
    @contextmanager
    def surcharge(self, **options):
        """Surcharge temporaire des options d'echantillonnage.

            with llm.surcharge(repeat_penalty=1.1, repeat_last_n=256):
                ...  # tous les appels de ce bloc portent ces options
        """
        precedente = self._surcharge
        self._surcharge = dict(precedente, **options)
        try:
            yield self
        finally:
            self._surcharge = precedente

    def call_llm(self, system_prompt: str, prompt: str) -> str:
        return self._call([{"role": "system", "content": system_prompt},
                           {"role": "user", "content": prompt}])

    def generate(self, query: str, context: str, search_results: list = None,
                 custom_prompt: str = None, history: list = None) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self._system_prompt(custom_prompt)}]
        for m in history or []:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": self._rag_prompt(context, query) if context else query})
        try:
            texte = self._call(messages)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Erreur generation LLM : {e}")
            texte = f"Erreur lors de la generation : {e}"
        return {"response": texte, "sources": self._sources(search_results),
                "confidence": 0.8 if context else 0.3, "model": self.model,
                "usage": dict(self.last_usage)}

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as c:
                return c.get(f"{self.base_url}/models").status_code == 200
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    def _options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        options.update(self.options_base)
        options.update(self._surcharge)
        return options

    def _call(self, messages: list) -> str:
        self.last_usage = {}
        options = self._options()
        if self.api_format == "ollama":
            payload = {"model": self.model, "messages": messages, "stream": False, "options": options}
            url, extraire = f"{self.native_base}/api/chat", lambda j: j["message"]["content"]
        else:
            payload = {"model": self.model, "messages": messages, "stream": False,
                       "temperature": self.temperature, "max_tokens": self.max_tokens}
            if self.seed is not None:
                payload["seed"] = self.seed
            for cle, cle_openai in _VERS_OPENAI.items():
                if cle in options:
                    payload[cle_openai] = options[cle]
            ignorees = sorted(k for k in (set(self.options_base) | set(self._surcharge))
                              if k not in _VERS_OPENAI)
            if ignorees:
                logger.debug(f"options sans equivalent OpenAI ignorees : {ignorees}")
            url, extraire = f"{self.base_url}/chat/completions", lambda j: j["choices"][0]["message"]["content"]
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                j = r.json()
                self._journaliser_tokens(j)
                return extraire(j)
        except httpx.TimeoutException:
            raise RuntimeError(f"LLM timeout ({self.timeout}s) sur {url}")
        except httpx.ConnectError:
            raise RuntimeError(f"LLM inaccessible ({url})")

    def _journaliser_tokens(self, j: Dict[str, Any]) -> None:
        """Occupation reelle de la fenetre, mesuree par le backend, et
        motif d'arret de la generation.

        Ollama (/api/chat) renvoie prompt_eval_count, eval_count et
        done_reason ("stop" | "length") ; l'API OpenAI-compatible renvoie
        usage.prompt_tokens / completion_tokens et choices[0].finish_reason.
        Le tout est conserve dans `last_usage` pour les gardes de query2.
        """
        try:
            usage = j.get("usage") or {}
            entree = j.get("prompt_eval_count", usage.get("prompt_tokens"))
            sortie = j.get("eval_count", usage.get("completion_tokens"))
            motif = j.get("done_reason")
            if motif is None:
                choix = (j.get("choices") or [{}])[0]
                motif = choix.get("finish_reason")
            tronque = (motif == "length") or bool(
                sortie is not None and self.max_tokens and sortie >= self.max_tokens)
            self.last_usage = {"prompt": entree, "generes": sortie,
                               "done_reason": motif, "tronque": tronque}
            if entree is None:
                return
            total = entree + (sortie or 0)
            ctx = self.num_ctx or 0
            pct = f" ({100 * total // ctx} % de num_ctx={ctx})" if ctx else ""
            message = f"LLM tokens: prompt={entree} generes={sortie} total={total}{pct} arret={motif}"
            if tronque:
                logger.warning(f"{message} — generation arretee par max_tokens={self.max_tokens}")
            elif ctx and total >= 0.9 * ctx:
                logger.warning(f"{message} — fenetre presque pleine, troncature possible")
            else:
                logger.info(message)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"comptage tokens indisponible ({e})")

    def _system_prompt(self, custom_prompt: Optional[str]) -> str:
        base = (self.prompts.get_system_prompt() if self.prompts else
                "Tu es Luciole, un assistant documentaire. Tu t'appuies sur les "
                "documents fournis pour repondre. Ne jamais inventer. Cite tes sources.")
        return f"{base}\n\n{custom_prompt}" if custom_prompt else base

    def _rag_prompt(self, context: str, query: str) -> str:
        if self.prompts:
            try:
                if (self.prompts.get_rag_prompt() or "").strip():
                    return self.prompts.format_rag_prompt(context, query)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"rag_prompt YAML inutilisable ({e})")
        return (f"Voici des extraits de documents pertinents :\n\n{context}\n\n---\n\n"
                f"Question : {query}\n\nReponds en t'appuyant exclusivement sur les extraits "
                "ci-dessus. Cite le document et la page de l'etiquette de source de chaque "
                "extrait utilise. Si l'information n'est pas presente, dis-le clairement.")

    @staticmethod
    def _sources(search_results: Optional[list]) -> List[Dict]:
        vus, sources = set(), []
        for r in search_results or []:
            meta = r.get("metadata") or {}
            nom = r.get("file_name") or meta.get("file_name", "")
            chemin = r.get("file_path") or meta.get("file_path", "")
            debut, fin = meta.get("page_start"), meta.get("page_end")
            cle = (chemin or nom, debut, fin)
            if not (chemin or nom) or cle in vus:
                continue
            vus.add(cle)
            sources.append({"file_name": nom, "file_path": chemin, "page_start": debut,
                            "page_end": fin, "score": round(r.get("rrf_score", r.get("score", 0)) or 0, 4)})
        return sources[:10]
