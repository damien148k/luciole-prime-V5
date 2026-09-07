"""Generateur LLM (Ollama natif ou OpenAI-compatible), configure par Settings.

Repris de V3 (`generation/llm.py`) : memes deux chemins d'appel, meme
prompt RAG lu dans prompts.yaml. Difference V4 : les reglages sont lus
sur l'objet Settings recu, jamais sur une relecture privee du YAML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from src.config.prompts import load_prompts
from src.config.settings import Settings


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
        self.prompts = prompts if prompts is not None else self._charger_prompts()
        logger.info(f"LLMGenerator : model={self.model}, base={base}, format={self.api_format}")

    @staticmethod
    def _charger_prompts():
        try:
            return load_prompts()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"prompts.yaml illisible ({e}) : gabarits integres")
            return None

    # ------------------------------------------------------------------
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
                "confidence": 0.8 if context else 0.3, "model": self.model}

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as c:
                return c.get(f"{self.base_url}/models").status_code == 200
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    def _call(self, messages: list) -> str:
        if self.api_format == "ollama":
            options = {"temperature": self.temperature, "num_predict": self.max_tokens}
            if self.seed is not None:
                options["seed"] = self.seed
            if self.num_ctx:
                options["num_ctx"] = self.num_ctx
            payload = {"model": self.model, "messages": messages, "stream": False, "options": options}
            url, extraire = f"{self.native_base}/api/chat", lambda j: j["message"]["content"]
        else:
            payload = {"model": self.model, "messages": messages, "stream": False,
                       "temperature": self.temperature, "max_tokens": self.max_tokens}
            if self.seed is not None:
                payload["seed"] = self.seed
            url, extraire = f"{self.base_url}/chat/completions", lambda j: j["choices"][0]["message"]["content"]
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                return extraire(r.json())
        except httpx.TimeoutException:
            raise RuntimeError(f"LLM timeout ({self.timeout}s) sur {url}")
        except httpx.ConnectError:
            raise RuntimeError(f"LLM inaccessible ({url})")

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
