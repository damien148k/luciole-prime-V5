"""Évaluation RAGAS 100% offline via TensorRT-LLM local. Aucun appel externe."""
import os
os.environ.setdefault("OPENAI_API_KEY", "not-needed")

import math
import re
import numpy as np
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
import sqlite3
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing import List, Optional


def _sanitize_scores(scores: dict) -> dict:
    """Replace NaN/Inf with None so JSON serialization works."""
    clean = {}
    for k, v in scores.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return clean


def _is_question(text: str) -> bool:
    """Detect if text is an actual question vs a statement/recommendation."""
    text = text.strip()
    if text.endswith("?"):
        return True
    q_patterns = [
        r"^(quel|quelle|quels|quelles|qu['e])\b",
        r"^(comment|pourquoi|combien|où|quand)\b",
        r"^(est-ce que|y a-t-il|peut-on|faut-il)\b",
        r"^(what|how|why|where|when|which|who|is there|can)\b",
    ]
    lower = text.lower()
    return any(re.search(p, lower) for p in q_patterns)


def _cosine_similarity(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return max(0.0, float(dot / norm))


class LucioleRAGASEvaluator:
    """
    Évaluateur RAGAS offline.

    Métriques :
      - faithfulness (RAGAS) : la réponse est-elle fidèle au contexte ?
      - answer_relevancy (calcul custom) : la réponse est-elle pertinente ?
        Implémentation propre car RAGAS 0.4.x a un bug de parsing avec les LLM locaux.
      - context_recall (RAGAS) : le contexte couvre-t-il la bonne info ? (nécessite ground_truth)
    """

    def __init__(self, model: str, llm_url: str = None, ollama_url: str = None, db_path: str = "feedbacks/ragas.db",
                 embed_model: str = "nomic-embed-text"):
        # Rétrocompatibilité : ollama_url accepté comme alias de llm_url
        if llm_url is None and ollama_url is not None:
            llm_url = ollama_url
        elif llm_url is None:
            llm_url = os.environ.get("LLM_URL", "http://tensorrt-llm:8000")
        self._chat_client = ChatOpenAI(
            base_url=f"{llm_url}/v1",
            api_key="not-needed",
            model=model,
            temperature=0,
            max_retries=3,
        )
        self._embed_client = OpenAIEmbeddings(
            base_url=f"{llm_url}/v1",
            api_key="not-needed",
            model=embed_model,
            check_embedding_ctx_length=False,
        )
        self.llm = LangchainLLMWrapper(self._chat_client)
        self.embeddings = LangchainEmbeddingsWrapper(self._embed_client)
        self.db_path = db_path
        self._init_db()

    def _reformulate_as_question(self, statement: str) -> str:
        """Use LLM to reformulate a statement/recommendation into a question."""
        try:
            response = self._chat_client.invoke(
                "Reformule cette affirmation en une question courte et directe "
                "(une seule phrase interrogative, en francais, terminee par '?'). "
                "Reponds UNIQUEMENT avec la question, rien d'autre.\n\n"
                f"Affirmation: {statement[:500]}"
            )
            reformulated = response.content.strip().strip('"').strip("'")
            if reformulated and reformulated.endswith("?"):
                logger.info(f"RAGAS reformulation: '{statement[:60]}...' -> '{reformulated[:80]}'")
                return reformulated
        except Exception as e:
            logger.warning(f"Reformulation failed: {e}")
        return statement

    def _compute_answer_relevancy(self, question: str, answer: str, n_generations: int = 3) -> Optional[float]:
        """
        Custom answer_relevancy: generate questions from the answer,
        embed them alongside the original question, compute cosine similarity.
        Replaces RAGAS's broken implementation for local LLMs.
        """
        try:
            generated_questions = []
            for i in range(n_generations):
                prompt = (
                    "Generate a short question in the same language as the answer below. "
                    "The question should be answerable by this answer. "
                    "Reply ONLY with the question, nothing else.\n\n"
                    f"Answer: {answer[:800]}"
                )
                resp = self._chat_client.invoke(prompt)
                gq = resp.content.strip().strip('"').strip("'")
                if gq:
                    generated_questions.append(gq)

            if not generated_questions:
                logger.warning("answer_relevancy: no questions generated")
                return None

            q_embedding = self._embed_client.embed_query(question[:500])
            similarities = []
            for gq in generated_questions:
                gq_embedding = self._embed_client.embed_query(gq[:500])
                sim = _cosine_similarity(q_embedding, gq_embedding)
                similarities.append(sim)
                logger.debug(f"answer_relevancy: sim={sim:.4f} gq='{gq[:60]}'")

            score = float(np.mean(similarities))
            logger.info(f"answer_relevancy (custom): {score:.4f} ({len(similarities)} questions)")
            return round(score, 4)
        except Exception as e:
            logger.warning(f"answer_relevancy computation failed: {e}")
            return None

    def evaluate_single(self, question: str, answer: str,
                        contexts: List[str], index_name: str,
                        ground_truth: Optional[str] = None) -> dict:
        if contexts:
            contexts = [c[:3000] for c in contexts[:5]]

        eval_question = question
        if not _is_question(question):
            eval_question = self._reformulate_as_question(question)

        # 1) Compute answer_relevancy with our custom method
        rel_score = self._compute_answer_relevancy(eval_question, answer)

        # 2) Run RAGAS for faithfulness (+ context_recall if ground_truth)
        data = {
            "question": [eval_question], "answer": [answer], "contexts": [contexts],
        }
        ragas_metrics = [faithfulness]
        if ground_truth:
            data["reference"] = [ground_truth]
            ragas_metrics.append(context_recall)

        dataset = Dataset.from_dict(data)
        try:
            result = evaluate(
                dataset, metrics=ragas_metrics,
                llm=self.llm, embeddings=self.embeddings,
                raise_exceptions=False,
            )
            scores = result.to_pandas().to_dict(orient="records")[0]
        except Exception as e:
            logger.warning(f"RAGAS evaluate() raised: {e}")
            scores = {"faithfulness": None, "context_recall": None}

        # 3) Merge custom answer_relevancy into scores
        scores["answer_relevancy"] = rel_score
        scores = _sanitize_scores(scores)
        self._store(question, scores, index_name, ground_truth)
        return scores

    def get_dashboard(self, index_name: str, days: int = 30) -> dict:
        """Métriques agrégées des N derniers jours."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    AVG(faithfulness) as avg_faith,
                    AVG(answer_relevancy) as avg_rel,
                    AVG(context_recall) as avg_recall,
                    COUNT(*) as total
                FROM ragas_scores
                WHERE index_name = ?
                AND timestamp >= datetime('now', ?)
            """, (index_name, f"-{days} days"))
            row = cursor.fetchone()
            if row and row[3] > 0:
                return {
                    "faithfulness": round(row[0], 3) if row[0] else None,
                    "answer_relevancy": round(row[1], 3) if row[1] else None,
                    "context_recall": round(row[2], 3) if row[2] else None,
                    "total_evaluations": row[3],
                    "period_days": days,
                    "index_name": index_name,
                }
            return {"total_evaluations": 0, "index_name": index_name}

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS ragas_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, index_name TEXT, question TEXT,
                faithfulness REAL, answer_relevancy REAL, context_recall REAL,
                ground_truth TEXT)""")
            try:
                conn.execute("SELECT ground_truth FROM ragas_scores LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE ragas_scores ADD COLUMN ground_truth TEXT")

    def _store(self, question: str, scores: dict, index_name: str,
               ground_truth: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO ragas_scores (timestamp, index_name, question, "
                "faithfulness, answer_relevancy, context_recall, ground_truth) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), index_name, question,
                 scores.get("faithfulness"), scores.get("answer_relevancy"),
                 scores.get("context_recall"), ground_truth))
