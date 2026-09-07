"""
Feedback UI - Interface de test pour Keyusers avec système de feedback
Permet aux testeurs d'évaluer les réponses de Luciole et de proposer des corrections.
V3 : Authentification cookie + onglet RAGAS
V4 : Intégration module mail (paramètres, tests, brouillons, santé)
"""

import os
import json
import time
import sqlite3
import csv
import io
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
from loguru import logger
from src.ui.auth import (
    verify_credentials, make_session_token, validate_session_token,
    get_login_html, AUTH_COOKIE_NAME,
)

# ── Module mail ────────────────────────────────────────────────────────────
# Service distinct de la V3 (luciole-mail), non repris en V4. Les pages
# /mail restent servies mais leurs appels de donnees echouent proprement.
_mail_available = False
logger.info("Module mail absent en V4 : les pages /mail restent inertes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan : démarrage du scheduler mail si disponible."""
    _mail_task = None
    if _mail_available:
        try:
            init_mail_tables()
            _mail_task = asyncio.create_task(start_mail_scheduler())
            logger.info("Scheduler mail démarré")
        except Exception as e:
            logger.error(f"Impossible de démarrer le scheduler mail : {e}")
    yield
    if _mail_task:
        _mail_task.cancel()
        try:
            await _mail_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Luciole Feedback",
    description="Interface de test Keyusers avec système de feedback",
    version="4.0.0",
    lifespan=lifespan,
)

# Montage du router mail (toutes les routes /api/mail/*)
if _mail_available:
    app.include_router(mail_router)

_PUBLIC_PATHS = {"/login", "/logout", "/logo.png", "/logo.svg", "/favicon.ico", "/favicon.png", "/health"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Vérifie le cookie de session sur toutes les routes sauf publiques."""
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)
    token = request.cookies.get(AUTH_COOKIE_NAME)
    username = validate_session_token(token)
    if not username:
        return HTMLResponse(get_login_html(), status_code=401)
    request.state.username = username
    return await call_next(request)


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if not verify_credentials(username, password):
        return HTMLResponse(get_login_html("Identifiants incorrects"), status_code=401)
    token = make_session_token(username)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(AUTH_COOKIE_NAME, token, max_age=86400, httponly=True, samesite="lax")
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# V4 : plus de conteneur agent. Le recueil d'avis parle au coeur RAG par
# son contrat public (POST /api/rag/query, POST /api/reload-config) et
# s'adresse a Ollama en direct pour la gestion des modeles — ce que la
# V3 faisait passer par l'agent, qui ne faisait que relayer.
RAG_URL = os.environ.get("RAG_URL") or os.environ.get("AGENT_URL", "http://rag:8000")
OLLAMA_URL = os.environ.get("LLM_URL", "http://ollama:11434")
SERVICE_NAME = os.environ.get("INSTANCE_NAME", "Luciole")
FEEDBACK_DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "/app/feedbacks/feedbacks.db")

# Chemin vers le logo
STATIC_DIR = Path(__file__).parent / "static"
LOGO_PATH = STATIC_DIR / "logo.png"
PICS_DIR = Path(__file__).parent.parent.parent / "pics"


# ============================================================================
# DATABASE SETUP
# ============================================================================

def get_db_connection():
    """Retourne une connexion à la base SQLite"""
    db_path = Path(FEEDBACK_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialise la base de données si elle n'existe pas"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            sources TEXT,
            index_name TEXT,
            feedback TEXT CHECK(feedback IN ('up', 'down')),
            expected_response TEXT,
            comment TEXT,
            processing_time_ms INTEGER
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Base de données initialisée: {FEEDBACK_DB_PATH}")


# Initialiser la DB au démarrage
init_database()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    index_name: Optional[str] = None
    top_k: int = 20
    custom_prompt: Optional[str] = None
    deep_search: bool = False
    history: list[ChatMessage] = []


class FeedbackRequest(BaseModel):
    query: str
    response: str
    sources: Optional[str] = None
    index_name: Optional[str] = None
    feedback: str  # "up" ou "down"
    expected_response: Optional[str] = None
    comment: Optional[str] = None
    processing_time_ms: Optional[int] = None
    user_id: Optional[str] = None


# ============================================================================
# STATIC FILES
# ============================================================================

@app.get("/logo.png")
async def get_logo():
    pics_logo = PICS_DIR / "luciole-logo.png"
    if pics_logo.exists():
        return FileResponse(pics_logo, media_type="image/png")
    if LOGO_PATH.exists():
        return FileResponse(LOGO_PATH, media_type="image/png")
    alt_path = PICS_DIR / "luciole.png"
    if alt_path.exists():
        return FileResponse(alt_path, media_type="image/png")
    return HTMLResponse("Logo not found", status_code=404)


@app.get("/favicon.ico")
@app.get("/favicon.png")
async def get_favicon():
    # Meme ordre que le Chat UI : petit PNG optimise pour l'onglet (favicon.png)
    for candidate in (
        PICS_DIR / "favicon.png",
        STATIC_DIR / "favicon.png",
        PICS_DIR / "luciole-logo.png",
        LOGO_PATH,
        PICS_DIR / "luciole.png",
    ):
        if candidate.exists():
            return FileResponse(candidate, media_type="image/png")
    return HTMLResponse("Favicon not found", status_code=404)


@app.get("/logo.svg")
async def get_logo_svg():
    svg_path = PICS_DIR / "luciole.svg"
    if svg_path.exists():
        return FileResponse(svg_path, media_type="image/svg+xml")
    return HTMLResponse("SVG logo not found", status_code=404)


# ============================================================================
# FEEDBACK API
# ============================================================================

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Enregistre un feedback dans la base de données."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedbacks (user_id, query, response, sources, index_name, 
                                   feedback, expected_response, comment, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.user_id,
            feedback.query,
            feedback.response,
            feedback.sources,
            feedback.index_name,
            feedback.feedback,
            feedback.expected_response,
            feedback.comment,
            feedback.processing_time_ms
        ))
        conn.commit()
        feedback_id = cursor.lastrowid
        conn.close()
        
        logger.info(f"Feedback enregistré #{feedback_id}: {feedback.feedback} par {feedback.user_id}")
        
        return {"status": "success", "id": feedback_id}
    except Exception as e:
        logger.error(f"Erreur enregistrement feedback: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/feedback/config")
async def feedback_config():
    """Config du recueil d'avis, lue par l'UI de chat.

    En V3 elle venait de l'agent ; en V4 le service la declare lui-meme,
    puisque c'est lui qui detient la base. Le service repond donc = les
    avis sont actifs.
    """
    return {"enabled": True, "key_users": []}


@app.get("/api/feedbacks")
async def get_feedbacks(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    feedback_type: Optional[str] = Query(None, pattern="^(up|down)$"),
    user_id: Optional[str] = None
):
    """Récupère la liste des feedbacks"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM feedbacks WHERE 1=1"
        params = []
        
        if feedback_type:
            query += " AND feedback = ?"
            params.append(feedback_type)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Compter le total
        count_query = "SELECT COUNT(*) FROM feedbacks WHERE 1=1"
        count_params = []
        if feedback_type:
            count_query += " AND feedback = ?"
            count_params.append(feedback_type)
        if user_id:
            count_query += " AND user_id = ?"
            count_params.append(user_id)
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()[0]
        
        conn.close()
        
        feedbacks = [dict(row) for row in rows]
        return {
            "feedbacks": feedbacks,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Erreur récupération feedbacks: {e}")
        return {"feedbacks": [], "total": 0, "error": str(e)}


@app.get("/api/feedbacks/stats")
async def get_feedback_stats():
    """Statistiques des feedbacks"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total
        cursor.execute("SELECT COUNT(*) FROM feedbacks")
        total = cursor.fetchone()[0]
        
        # Par type
        cursor.execute("SELECT feedback, COUNT(*) FROM feedbacks GROUP BY feedback")
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Par user
        cursor.execute("SELECT user_id, COUNT(*) FROM feedbacks GROUP BY user_id ORDER BY COUNT(*) DESC LIMIT 10")
        by_user = [{"user_id": row[0] or "anonyme", "count": row[1]} for row in cursor.fetchall()]
        
        # Récents (7 derniers jours)
        cursor.execute("""
            SELECT DATE(timestamp) as date, feedback, COUNT(*) 
            FROM feedbacks 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp), feedback
            ORDER BY date
        """)
        recent = [{"date": row[0], "feedback": row[1], "count": row[2]} for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "total": total,
            "up": by_type.get("up", 0),
            "down": by_type.get("down", 0),
            "by_user": by_user,
            "recent": recent
        }
    except Exception as e:
        logger.error(f"Erreur stats: {e}")
        return {"error": str(e)}


@app.get("/api/feedbacks/export")
async def export_feedbacks_csv():
    """Export des feedbacks au format CSV"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedbacks ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        
        # Créer le CSV en mémoire
        output = io.StringIO()
        writer = csv.writer(output, delimiter=';', quoting=csv.QUOTE_ALL)
        
        # En-têtes
        writer.writerow(['ID', 'Timestamp', 'User ID', 'Question', 'Réponse', 
                        'Sources', 'Index', 'Feedback', 'Correction attendue', 
                        'Commentaire', 'Temps (ms)'])
        
        # Données
        for row in rows:
            writer.writerow([
                row['id'],
                row['timestamp'],
                row['user_id'] or '',
                row['query'],
                row['response'],
                row['sources'] or '',
                row['index_name'] or '',
                row['feedback'],
                row['expected_response'] or '',
                row['comment'] or '',
                row['processing_time_ms'] or ''
            ])
        
        output.seek(0)
        
        filename = f"feedbacks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Erreur export CSV: {e}")
        return {"error": str(e)}


# ============================================================================
# PROXY API (vers l'agent)
# ============================================================================


# ============================================================================
# MODÈLE LLM ACTIF + gestion dynamique (Ollama)
# ============================================================================
# Ces proxies transfèrent vers l'Agent API. La gestion dynamique n'est active
# que si le backend le supporte (Ollama) ; sur TensorRT-LLM (ou tout autre
# backend non-Ollama) l'agent renvoie HTTP 501, transféré tel quel à l'UI.

@app.get("/api/llm/model")
async def llm_model_actif():
    """Modele LLM actif et capacites du backend.

    Lu dans settings.yaml (source de verite unique de l'instance) plutot
    que demande a l'agent : en V4 il n'y a plus d'agent, et le coeur RAG
    n'expose pas ce reglage.
    """
    modele, backend = "", "ollama"
    try:
        import yaml
        with open(CONFIG_FILES["settings.yaml"], encoding="utf-8") as f:
            llm = (yaml.safe_load(f) or {}).get("llm", {}) or {}
        modele = llm.get("model", "")
        backend = llm.get("provider", "ollama")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"settings.yaml illisible pour llm.model ({e})")
    return {"model": modele, "backend": backend,
            "supports_hot_swap": backend == "ollama"}


@app.get("/api/ollama/models")
async def ollama_modeles():
    """Modeles installes (API native Ollama : GET /api/tags)."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            data = (await client.get(f"{OLLAMA_URL}/api/tags")).json()
        return {"models": [
            {"name": m.get("name", ""), "size": m.get("size", 0),
             "modified_at": m.get("modified_at", "")}
            for m in data.get("models", [])
        ]}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Ollama injoignable ({e})")
        return JSONResponse(status_code=502, content={"error": f"Ollama injoignable: {e}"})


@app.post("/api/ollama/pull")
async def ollama_telecharger(request: Request):
    """Telechargement d'un modele, progression relayee au fil de l'eau."""
    body = await request.body()

    async def relai():
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30, read=3600, write=30, pool=30)
            ) as client:
                async with client.stream(
                    "POST", f"{OLLAMA_URL}/api/pull", content=body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread()).decode(errors="ignore")
                        yield f"data: {json.dumps({'error': detail})}\n\n"
                        return
                    async for ligne in resp.aiter_lines():
                        if ligne:
                            yield f"data: {ligne}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.error(f"Telechargement Ollama en echec : {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(relai(), media_type="text/event-stream")


@app.post("/api/ollama/activate")
async def ollama_activer(request: Request):
    """Rend un modele installe actif pour l'instance.

    Ecrit llm.model dans settings.yaml puis demande au coeur RAG de
    relire sa configuration : le generateur LLM est reconstruit a chaud,
    l'embedder et le reranker sont conserves.
    """
    corps = await request.json()
    modele = str(corps.get("model", "")).strip()
    if not modele:
        return JSONResponse(status_code=400, content={"error": "Modele non precise"})
    try:
        import yaml
        chemin = CONFIG_FILES["settings.yaml"]
        with open(chemin, encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        conf.setdefault("llm", {})["model"] = modele
        with open(chemin, "w", encoding="utf-8") as f:
            yaml.safe_dump(conf, f, allow_unicode=True, sort_keys=False)
        async with httpx.AsyncClient(timeout=30.0) as client:
            rechargement = (await client.post(f"{RAG_URL}/api/reload-config")).json()
        logger.info(f"Modele actif : {modele}")
        return {"status": "ok", "model": modele, "reload": rechargement}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Activation du modele en echec : {e}")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/ollama/search")
async def ollama_recherche(q: str = ""):
    """Recherche au registre : indisponible sur une instance hors ligne.

    La V3 la faisait passer par l'agent, qui interrogeait ollama.com.
    Une instance V4 est concue pour tourner sans acces internet : plutot
    qu'echouer par un timeout, on le dit franchement a l'interface.
    """
    return JSONResponse(status_code=501, content={
        "error": "Recherche au registre indisponible (instance hors ligne). "
                 "Telechargez un modele par son nom exact.",
        "models": [],
    })


@app.delete("/api/ollama/models")
async def ollama_supprimer(request: Request):
    """Suppression d'un modele installe (API native : DELETE /api/delete)."""
    corps = await request.json()
    modele = str(corps.get("model", "")).strip()
    if not modele:
        return JSONResponse(status_code=400, content={"error": "Modele non precise"})
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.request(
                "DELETE", f"{OLLAMA_URL}/api/delete",
                json={"model": modele}, headers={"Content-Type": "application/json"})
        if resp.status_code >= 400:
            return JSONResponse(status_code=resp.status_code,
                                content={"error": resp.text[:200]})
        return {"status": "ok", "model": modele}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Suppression du modele en echec : {e}")
        return JSONResponse(status_code=502, content={"error": str(e)})


CONFIG_FILES = {
    "settings.yaml": "/app/config/settings.yaml",
    "prompts.yaml": "/app/config/prompts.yaml",
    "synonyms.txt": "/app/config/synonyms.txt",
}

READONLY_FILES = {}


class ConfigSaveRequest(BaseModel):
    filename: str
    content: str


@app.get("/api/config/{filename}")
async def read_config_file(filename: str):
    """Lit le contenu d'un fichier de configuration"""
    all_files = {**CONFIG_FILES, **READONLY_FILES}
    if filename not in all_files:
        return {"error": f"Fichier inconnu: {filename}", "files": list(all_files.keys())}
    
    filepath = all_files[filename]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "filename": filename,
            "content": content,
            "readonly": filename in READONLY_FILES,
            "path": filepath
        }
    except FileNotFoundError:
        return {"error": f"Fichier non trouvé: {filepath}", "content": ""}
    except Exception as e:
        return {"error": str(e), "content": ""}


@app.post("/api/config/save")
async def save_config_file(request: ConfigSaveRequest):
    """Sauvegarde un fichier de configuration"""
    if request.filename not in CONFIG_FILES:
        return {"error": f"Fichier non modifiable: {request.filename}"}
    
    filepath = CONFIG_FILES[request.filename]
    try:
        # Validation YAML pour les fichiers .yaml
        if request.filename.endswith(".yaml"):
            import yaml
            yaml.safe_load(request.content)  # Valide la syntaxe
        
        # Validation syntaxe Python pour les fichiers .py
        if request.filename.endswith(".py"):
            try:
                compile(request.content, request.filename, "exec")
            except SyntaxError as e:
                return {"status": "error", "message": f"Erreur syntaxe Python ligne {e.lineno}: {e.msg}"}
        
        # Sauvegarder
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(request.content)
        
        logger.info(f"Config saved: {request.filename}")
        return {"status": "ok", "message": f"{request.filename} sauvegardé"}
    except yaml.YAMLError as e:
        return {"status": "error", "message": f"Erreur YAML: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/config/reload")
async def reload_agent_config():
    """Demande à l'agent de recharger sa configuration à chaud"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(f"{RAG_URL}/api/reload-config")
            result = response.json()
            logger.info(f"Agent config reload: {result}")
            return result
    except Exception as e:
        logger.error(f"Error reloading agent config: {e}")
        return {"status": "error", "message": f"Erreur communication agent: {e}"}


@app.get("/api/config/files")
async def list_config_files():
    """Liste les fichiers de configuration disponibles"""
    files = []
    for name, path in CONFIG_FILES.items():
        files.append({"name": name, "path": path, "editable": True})
    for name, path in READONLY_FILES.items():
        files.append({"name": name, "path": path, "editable": False})
    return {"files": files}


@app.get("/api/indexes")
async def get_indexes():
    """V4 : une instance = un index. Le nom vient du coeur RAG."""
    nom = SERVICE_NAME
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            nom = (await client.get(f"{RAG_URL}/api/health")).json().get("instance") or nom
    except Exception as e:  # noqa: BLE001
        logger.warning(f"/api/health injoignable, nom par defaut ({e})")
    return {"indexes": [nom], "default": nom, "index": nom,
            "instance_name": nom, "single_index_mode": True}


@app.post("/api/query")
async def query(request: ChatRequest):
    """Proxy vers le coeur RAG, meme traduction que l'UI de chat.

    query -> question, deep_search -> deep ; answer -> response,
    citations -> sources. index_name et top_k disparaissent : une
    instance V4 = un index, l'entonnoir vient de settings.yaml.
    """
    debut = time.perf_counter()
    try:
        history_list = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
        async with httpx.AsyncClient(timeout=1800.0) as client:
            reponse = await client.post(f"{RAG_URL}/api/rag/query", json={
                "question": request.query,
                "custom_prompt": request.custom_prompt,
                "deep": request.deep_search,
                "history": history_list,
            })
        ecoule = int((time.perf_counter() - debut) * 1000)
        if reponse.status_code != 200:
            return {"error": f"HTTP {reponse.status_code}",
                    "response": f"Le coeur RAG a renvoye une erreur {reponse.status_code}.",
                    "detail": reponse.text[:300], "processing_time_ms": ecoule}
        d = reponse.json()
        return {
            "response": d.get("answer", ""),
            "sources": d.get("citations", []),
            "passages": d.get("passages", []),
            "query": request.query,
            "index": SERVICE_NAME,
            "processing_time_ms": ecoule,
            "verdict": d.get("verdict"),
            "confidence": d.get("confidence"),
            "esquive": d.get("esquive"),
            "trace_id": d.get("trace_id"),
            "trace": d.get("trace", {}),
            "model": d.get("model", ""),
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Erreur d'appel au coeur RAG : {e}")
        return {"error": str(e), "response": f"Erreur: {e}",
                "processing_time_ms": int((time.perf_counter() - debut) * 1000)}


# ============================================================================
# MAIN CHAT PAGE (avec feedback)
# ============================================================================

@app.get("/")
async def home_redirect():
    """Redirige vers la page de configuration (le chat est desormais dans le Chat UI)."""
    return RedirectResponse(url="/config", status_code=303)


@app.get("/chat-legacy", response_class=HTMLResponse)
async def feedback_chat_page(user_id: str = Query(None, description="Identifiant du testeur")):
    """Page de chat legacy (conservee pour reference, redirigee depuis /)."""
    service_suffix = f" {SERVICE_NAME}" if SERVICE_NAME else ""
    page_title = f"Luciole Feedback{service_suffix}"
    user_display = user_id or "Anonyme"
    
    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{PAGE_TITLE}}</title>
    <link rel="icon" type="image/png" href="/favicon.png">
    <!-- Fonts: utilise les polices systeme (100% offline) -->
    <style>
        :root {
            --bg-primary: #0B1929;
            --bg-secondary: #0F2237;
            --bg-tertiary: #163050;
            --accent: #FFD76F;
            --accent-dim: #C4952C;
            --accent-glow: rgba(255, 215, 111, 0.25);
            --text-primary: #F8F7F1;
            --text-secondary: #7B96B2;
            --border: #1E3A56;
            --success: #34D399;
            --error: #F87171;
            --warning: #FFD76F;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(160deg, #070E18 0%, var(--bg-primary) 40%, #0D1F35 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(15, 34, 55, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 215, 111, 0.15);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
        }
        
        .logo { display: flex; align-items: center; gap: 0.75rem; }
        .logo-icon { width: 40px; height: 40px; }
        .logo-icon img { width: 100%; height: 100%; object-fit: contain; filter: drop-shadow(0 0 6px rgba(255,215,111,0.25)); }
        
        .logo h1 {
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, #FFD76F, #FFF0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header-right { display: flex; align-items: center; gap: 1rem; }
        
        .user-badge {
            background: var(--accent-dim);
            color: var(--accent);
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .dashboard-link {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            text-decoration: none;
            font-size: 0.85rem;
            transition: all 0.2s;
        }
        
        .dashboard-link:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        
        .new-chat-btn {
            background: linear-gradient(135deg, var(--accent), #FFCE60);
            border: none;
            color: #0B1929;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .new-chat-btn:hover { transform: scale(1.02); box-shadow: 0 2px 10px var(--accent-glow); }
        
        .index-selector { display: flex; align-items: center; gap: 0.75rem; }
        .index-selector label { color: var(--text-secondary); font-size: 0.9rem; }
        .index-selector select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: inherit;
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            padding: 1rem;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message { display: flex; gap: 1rem; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        
        .message.user { flex-direction: row-reverse; }
        
        .message-avatar {
            width: 40px; height: 40px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem; flex-shrink: 0;
        }
        
        .message.user .message-avatar { background: var(--accent); }
        .message.assistant .message-avatar { background: var(--bg-tertiary); border: 1px solid var(--border); }
        
        .message-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            max-width: 70%;
        }
        
        .message.user .message-content { background: var(--accent-dim); border-color: var(--accent); }
        
        .message-text { line-height: 1.6; white-space: pre-wrap; }
        
        .avatar-logo { height: 24px; width: auto; filter: drop-shadow(0 0 6px rgba(255,215,111,0.25)); }
        .welcome-logo { height: 80px; width: auto; margin-bottom: 1rem; filter: drop-shadow(0 0 6px rgba(255,215,111,0.25)); }
        
        /* FEEDBACK BUTTONS */
        .feedback-buttons {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
        }
        
        .feedback-btn {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 1.1rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .feedback-btn:hover { transform: scale(1.05); }
        .feedback-btn.up:hover { border-color: var(--success); background: rgba(16, 185, 129, 0.1); }
        .feedback-btn.down:hover { border-color: var(--error); background: rgba(239, 68, 68, 0.1); }
        .feedback-btn.selected.up { background: var(--success); border-color: var(--success); }
        .feedback-btn.selected.down { background: var(--error); border-color: var(--error); }
        .feedback-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .feedback-status {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-left: auto;
        }
        
        /* MODAL */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .modal-overlay.active { display: flex; }
        
        .modal {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal h2 {
            color: var(--error);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .modal-field { margin-bottom: 1rem; }
        .modal-field label { display: block; margin-bottom: 0.5rem; color: var(--text-secondary); font-size: 0.9rem; }
        
        .modal-field textarea, .modal-field input {
            width: 100%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
        }
        
        .modal-field textarea { min-height: 120px; resize: vertical; }
        .modal-field textarea:focus, .modal-field input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
        
        .modal-context {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-size: 0.85rem;
            max-height: 150px;
            overflow-y: auto;
        }
        
        .modal-context strong { color: var(--accent); }
        
        .modal-buttons { display: flex; gap: 1rem; justify-content: flex-end; }
        
        .modal-btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-family: inherit;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .modal-btn.primary { background: linear-gradient(135deg, var(--accent), #FFCE60); color: #0B1929; }
        .modal-btn.primary:hover { box-shadow: 0 2px 10px var(--accent-glow); }
        .modal-btn.secondary { background: var(--bg-tertiary); color: var(--text-secondary); border: 1px solid var(--border); }
        .modal-btn.secondary:hover { border-color: var(--text-secondary); }
        
        /* Sources */
        .message-sources { margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid var(--border); }
        .sources-toggle { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; color: var(--text-secondary); cursor: pointer; }
        .sources-toggle:hover { color: var(--accent); }
        .sources-toggle .toggle-icon { transition: transform 0.2s ease; }
        .sources-toggle.expanded .toggle-icon { transform: rotate(90deg); }
        .sources-list { display: none; margin-top: 0.5rem; padding-left: 1rem; }
        .sources-list.visible { display: block; }
        .source-item { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--accent); padding: 0.25rem 0; }
        
        .passages-section { margin-top: 0.75rem; border-top: 1px solid var(--border); padding-top: 0.75rem; }
        .passages-toggle { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; color: var(--text-secondary); cursor: pointer; }
        .passages-toggle:hover { color: var(--accent); }
        .passages-toggle .toggle-icon { transition: transform 0.2s ease; }
        .passages-toggle.expanded .toggle-icon { transform: rotate(90deg); }
        .passages-list { display: none; margin-top: 0.5rem; max-height: 400px; overflow-y: auto; }
        .passages-list.visible { display: block; }
        .passage-item { background: rgba(255,220,100,0.08); border-left: 3px solid #f0c040; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem; border-radius: 0 6px 6px 0; font-size: 0.8rem; line-height: 1.5; }
        .passage-meta { font-size: 0.7rem; color: var(--accent); margin-bottom: 0.3rem; font-weight: 600; }
        .passage-text { color: var(--text-secondary); white-space: pre-wrap; word-break: break-word; }
        
        .message-meta { margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-secondary); }
        
        /* Input */
        .input-area { background: var(--bg-secondary); border-top: 1px solid var(--border); padding: 1rem; }
        .input-wrapper { display: flex; gap: 0.75rem; max-width: 900px; margin: 0 auto; }
        .input-field {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            min-height: 50px;
            max-height: 150px;
        }
        .input-field:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
        .send-btn {
            background: linear-gradient(135deg, var(--accent), #FFCE60);
            border: none;
            border-radius: 12px;
            padding: 0 1.5rem;
            color: #0B1929;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        .send-btn:hover { box-shadow: 0 2px 10px var(--accent-glow); transform: scale(1.02); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        /* Loading */
        .loading { display: flex; gap: 0.5rem; padding: 1rem; }
        .loading-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
        .loading-dot:nth-child(1) { animation-delay: -0.32s; }
        .loading-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        
        /* Welcome */
        .welcome { text-align: center; padding: 3rem; color: var(--text-secondary); }
        .welcome h2 { color: var(--text-primary); margin-bottom: 0.5rem; }
        
        .test-banner {
            background: linear-gradient(135deg, var(--accent-dim), var(--accent));
            color: white;
            text-align: center;
            padding: 0.5rem;
            font-size: 0.85rem;
        }
        
        /* Sidebar */
        .sidebar-toggle {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.2rem;
            transition: all 0.2s;
        }
        .sidebar-toggle:hover { border-color: var(--accent); }
        .sidebar-toggle.active { background: var(--accent); color: white; border-color: var(--accent); }
        
        .sidebar {
            position: fixed;
            top: 0; right: -400px;
            width: 380px; height: 100vh;
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border-left: 1px solid var(--border);
            transition: right 0.3s ease;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            box-shadow: -5px 0 20px rgba(0, 0, 0, 0.3);
        }
        .sidebar.open { right: 0; }
        
        .sidebar-header {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sidebar-header h3 { font-size: 1.1rem; color: var(--accent); }
        .sidebar-close { background: none; border: none; color: var(--text-secondary); font-size: 1.5rem; cursor: pointer; }
        .sidebar-close:hover { color: var(--text-primary); }
        
        .sidebar-content { flex: 1; padding: 1rem; overflow-y: auto; }
        .sidebar-section { margin-bottom: 1.5rem; }
        .sidebar-section label { display: block; margin-bottom: 0.5rem; color: var(--text-secondary); font-size: 0.85rem; }
        .sidebar-section textarea {
            width: 100%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            padding: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            resize: vertical;
            min-height: 150px;
        }
        .sidebar-section textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
        .sidebar-section textarea:disabled { opacity: 0.5; }
        
        .toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-bottom: 0.75rem;
        }
        .toggle-switch { position: relative; width: 48px; height: 26px; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background: var(--border); border-radius: 26px; transition: 0.3s;
        }
        .toggle-slider:before {
            position: absolute; content: "";
            height: 20px; width: 20px; left: 3px; bottom: 3px;
            background: var(--text-primary); border-radius: 50%; transition: 0.3s;
        }
        .toggle-switch input:checked + .toggle-slider { background: var(--accent); }
        .toggle-switch input:checked + .toggle-slider:before { transform: translateX(22px); }
        
        .sidebar-footer { padding: 1rem; border-top: 1px solid var(--border); }
        .sidebar-btn {
            width: 100%; padding: 0.75rem;
            background: var(--accent); border: none; border-radius: 8px;
            color: white; font-family: inherit; font-weight: 500; cursor: pointer;
        }
        .sidebar-btn:hover { opacity: 0.9; }
        .sidebar-btn.secondary { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); margin-top: 0.5rem; }
        .sidebar-btn.secondary:hover { border-color: var(--accent); color: var(--accent); }
        
        .sidebar-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.5); z-index: 999; }
        .sidebar-overlay.active { display: block; }
    </style>
</head>
<body>
    <div class="test-banner">
        🧪 Mode Test - Évaluez les réponses avec 👍 ou 👎 pour améliorer Luciole
    </div>
    
    <header class="header">
        <div class="logo">
            <span class="logo-icon"><img src="/logo.png" alt="Luciole"></span>
            <h1>{{PAGE_TITLE}}</h1>
        </div>
        
        <div class="header-right">
            <div class="index-selector">
                <label for="index">Index:</label>
                <select id="index"><option value="">Chargement...</option></select>
            </div>
            <span class="user-badge">👤 {{USER_ID}}</span>
            <a href="/feedbacks" class="dashboard-link">📊 Dashboard</a>
            <a href="/config" class="dashboard-link">⚙️ Config</a>
            <a href="/mail" class="dashboard-link" id="mailNavLink">📧 Mail<span id="mailDraftsBadge" style="display:none;background:#ef4444;color:#fff;border-radius:10px;padding:1px 6px;font-size:0.7rem;margin-left:4px;vertical-align:middle"></span></a>
            <button class="new-chat-btn" onclick="startNewChat()" title="Démarrer une nouvelle conversation">✨ Nouveau chat</button>
            <button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" title="Paramètres avancés">⚙️</button>
        </div>
    </header>
    
    <!-- Sidebar Overlay -->
    <div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
    
    <!-- Sidebar Paramètres -->
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h3>⚙️ Paramètres avancés</h3>
            <button class="sidebar-close" onclick="closeSidebar()">×</button>
        </div>
        
        <div class="sidebar-content">
            <div class="sidebar-section">
                <div class="toggle-row">
                    <span>🎯 Prompt personnalisé</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="enableCustomPrompt" onchange="toggleCustomPrompt()">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                
                <textarea 
                    id="customPrompt" 
                    placeholder="Entrez votre prompt système personnalisé...

Exemple:
Tu es l'assistant du service juridique. Réponds de manière précise et cite les sources."
                    disabled
                ></textarea>
            </div>
            
            <div class="sidebar-section">
                <label>📊 Nombre de sources (top_k)</label>
                <select id="topKSelect" style="width: 100%; padding: 0.5rem; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 8px; color: var(--text-primary);">
                    <option value="5">5 sources</option>
                    <option value="10">10 sources</option>
                    <option value="15">15 sources</option>
                    <option value="20" selected>20 sources</option>
                </select>
            </div>
            
            <div class="sidebar-section">
                <div class="toggle-row">
                    <span>🔍 Recherche approfondie</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="enableDeepSearch">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <p style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;">
                    Double recherche avec/sans historique pour améliorer la précision. ⚠️ Temps de réponse ~2x plus long.
                </p>
            </div>
        </div>
        
        <div class="sidebar-footer">
            <button class="sidebar-btn" onclick="saveSettings()">💾 Sauvegarder</button>
            <button class="sidebar-btn secondary" onclick="resetSettings()">🔄 Réinitialiser</button>
        </div>
    </aside>
    
    <main class="chat-container">
        <div class="messages" id="messages">
            <div class="welcome">
                <img src="/logo.png" alt="Luciole" class="welcome-logo">
                <h2>Bienvenue sur {{PAGE_TITLE}}</h2>
                <p>Testez Luciole et évaluez ses réponses</p>
                <p style="margin-top: 1rem; font-size: 0.9rem;">
                    👍 Réponse satisfaisante &nbsp;|&nbsp; 👎 Réponse à améliorer
                </p>
            </div>
        </div>
    </main>
    
    <footer class="input-area">
        <div class="input-wrapper">
            <textarea class="input-field" id="messageInput" placeholder="Posez votre question..." rows="1"></textarea>
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Envoyer</button>
        </div>
    </footer>
    
    <!-- Modal de correction -->
    <div class="modal-overlay" id="feedbackModal">
        <div class="modal">
            <h2>👎 Signaler un problème</h2>
            
            <div class="modal-context" id="modalContext">
                <strong>Question :</strong> <span id="modalQuery"></span><br><br>
                <strong>Réponse actuelle :</strong> <span id="modalResponse"></span>
            </div>
            
            <div class="modal-field">
                <label>Quelle aurait été la bonne réponse ? *</label>
                <textarea id="expectedResponse" placeholder="Décrivez la réponse attendue ou corrigée..."></textarea>
            </div>
            
            <div class="modal-field">
                <label>Commentaire (optionnel)</label>
                <input type="text" id="feedbackComment" placeholder="Ex: Information manquante, source incorrecte...">
            </div>
            
            <div class="modal-buttons">
                <button class="modal-btn secondary" onclick="closeModal()">Annuler</button>
                <button class="modal-btn primary" onclick="submitDownFeedback()">Envoyer le feedback</button>
            </div>
        </div>
    </div>
    
    <script>
        const AGENT_URL = '/api';
        const USER_ID = '{{USER_ID_RAW}}';
        let isLoading = false;
        let conversationHistory = [];
        let currentFeedbackData = null;
        let messageCounter = 0;  // Compteur unique pour les IDs
        const feedbackDataStore = {};  // Stockage des données de feedback par ID
        
        // Fonction pour échapper le HTML et éviter les injections
        function escapeHtml(text) {
            if (!text) return '';
            return String(text)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;')
                .replace(/`/g, '&#096;')
                .replace(/\\$/g, '&#036;');
        }
        
        // Fonction pour formater le texte avec retours à la ligne
        function formatResponse(text) {
            if (!text) return '';
            return escapeHtml(text).replace(/\\n/g, '<br>');
        }
        
        // ========== SIDEBAR FUNCTIONS ==========
        
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebarOverlay');
            const toggle = document.getElementById('sidebarToggle');
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
            toggle.classList.toggle('active');
        }
        
        function closeSidebar() {
            document.getElementById('sidebar').classList.remove('open');
            document.getElementById('sidebarOverlay').classList.remove('active');
            document.getElementById('sidebarToggle').classList.remove('active');
        }
        
        function toggleCustomPrompt() {
            const enabled = document.getElementById('enableCustomPrompt').checked;
            document.getElementById('customPrompt').disabled = !enabled;
        }
        
        function getSettings() {
            const enableCustomPrompt = document.getElementById('enableCustomPrompt').checked;
            return {
                topK: parseInt(document.getElementById('topKSelect').value) || 20,
                customPrompt: enableCustomPrompt ? document.getElementById('customPrompt').value.trim() : null,
                deepSearch: document.getElementById('enableDeepSearch').checked
            };
        }
        
        function saveSettings() {
            const settings = getSettings();
            localStorage.setItem('feedbackSettings', JSON.stringify({
                topK: settings.topK,
                enableCustomPrompt: document.getElementById('enableCustomPrompt').checked,
                customPrompt: document.getElementById('customPrompt').value,
                deepSearch: settings.deepSearch
            }));
            showNotification('⚙️ Paramètres sauvegardés');
            closeSidebar();
        }
        
        function resetSettings() {
            document.getElementById('topKSelect').value = '20';
            document.getElementById('enableCustomPrompt').checked = false;
            document.getElementById('customPrompt').value = '';
            document.getElementById('customPrompt').disabled = true;
            document.getElementById('enableDeepSearch').checked = false;
            localStorage.removeItem('feedbackSettings');
            showNotification('🔄 Paramètres réinitialisés');
        }
        
        function loadSettings() {
            const saved = localStorage.getItem('feedbackSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                document.getElementById('topKSelect').value = settings.topK || '20';
                document.getElementById('enableCustomPrompt').checked = settings.enableCustomPrompt || false;
                document.getElementById('customPrompt').value = settings.customPrompt || '';
                document.getElementById('customPrompt').disabled = !settings.enableCustomPrompt;
                document.getElementById('enableDeepSearch').checked = settings.deepSearch || false;
            }
        }
        
        function startNewChat() {
            // Effacer l'historique de conversation
            conversationHistory = [];
            
            // Effacer les messages affichés
            const messages = document.getElementById('messages');
            messages.innerHTML = `
                <div class="welcome">
                    <img src="/logo.png" alt="Luciole" class="welcome-logo">
                    <h2>Bienvenue sur {{PAGE_TITLE}}</h2>
                    <p>Testez Luciole et évaluez ses réponses</p>
                    <p style="margin-top: 1rem; font-size: 0.9rem;">
                        👍 Réponse satisfaisante &nbsp;|&nbsp; 👎 Réponse à améliorer
                    </p>
                </div>
            `;
            
            showNotification('✨ Nouvelle conversation démarrée');
        }
        
        // ========== FEEDBACK FUNCTIONS ==========
        
        function handleFeedback(feedbackId, feedbackType) {
            // Récupérer les données depuis le store JavaScript
            const data = feedbackDataStore[feedbackId];
            
            if (!data) {
                console.error('Données de feedback non trouvées pour:', feedbackId);
                showNotification('❌ Erreur: données de feedback non trouvées');
                return;
            }
            
            submitFeedback(
                data.messageId,
                feedbackType,
                data.query,
                data.response,
                data.sources,
                data.indexName,
                data.processingTime
            );
        }
        
        async function submitFeedback(messageId, feedbackType, query, response, sources, indexName, processingTime) {
            if (feedbackType === 'down') {
                // Ouvrir le modal pour collecter plus d'infos
                currentFeedbackData = { messageId, query, response, sources, indexName, processingTime };
                document.getElementById('modalQuery').textContent = query.substring(0, 200) + (query.length > 200 ? '...' : '');
                document.getElementById('modalResponse').textContent = response.substring(0, 300) + (response.length > 300 ? '...' : '');
                document.getElementById('feedbackModal').classList.add('active');
                return;
            }
            
            // Feedback positif direct
            await sendFeedbackToServer({
                query,
                response,
                sources: JSON.stringify(sources),
                index_name: indexName,
                feedback: 'up',
                processing_time_ms: processingTime,
                user_id: USER_ID
            });
            
            // Marquer comme fait
            markFeedbackDone(messageId, 'up');
        }
        
        async function submitDownFeedback() {
            const expectedResponse = document.getElementById('expectedResponse').value.trim();
            if (!expectedResponse) {
                alert('Veuillez décrire la réponse attendue');
                return;
            }
            
            const comment = document.getElementById('feedbackComment').value.trim();
            
            await sendFeedbackToServer({
                query: currentFeedbackData.query,
                response: currentFeedbackData.response,
                sources: JSON.stringify(currentFeedbackData.sources),
                index_name: currentFeedbackData.indexName,
                feedback: 'down',
                expected_response: expectedResponse,
                comment: comment,
                processing_time_ms: currentFeedbackData.processingTime,
                user_id: USER_ID
            });
            
            markFeedbackDone(currentFeedbackData.messageId, 'down');
            closeModal();
        }
        
        async function sendFeedbackToServer(data) {
            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                if (result.status === 'success') {
                    showNotification(data.feedback === 'up' ? '👍 Merci pour votre feedback !' : '👎 Feedback enregistré, merci !');
                }
            } catch (e) {
                console.error('Erreur envoi feedback:', e);
                showNotification('Erreur lors de l\\'envoi du feedback');
            }
        }
        
        function markFeedbackDone(messageId, type) {
            const container = document.getElementById('feedback-' + messageId);
            if (container) {
                const buttons = container.querySelectorAll('.feedback-btn');
                buttons.forEach(btn => {
                    btn.disabled = true;
                    if (btn.classList.contains(type)) {
                        btn.classList.add('selected');
                    }
                });
                const status = container.querySelector('.feedback-status');
                if (status) {
                    status.textContent = type === 'up' ? '✓ Merci !' : '✓ Feedback enregistré';
                }
            }
        }
        
        function closeModal() {
            document.getElementById('feedbackModal').classList.remove('active');
            document.getElementById('expectedResponse').value = '';
            document.getElementById('feedbackComment').value = '';
            currentFeedbackData = null;
        }
        
        // ========== CHAT FUNCTIONS ==========
        
        async function loadIndexes() {
            try {
                const response = await fetch(`${AGENT_URL}/indexes`);
                const data = await response.json();
                const select = document.getElementById('index');
                select.innerHTML = '';

                // Mode mono-instance : 1 instance = 1 metier = 1 index
                if (data.single_index_mode && data.instance_name) {
                    const option = document.createElement('option');
                    option.value = data.instance_name;
                    option.textContent = data.instance_name;
                    select.appendChild(option);
                    select.value = data.instance_name;
                    select.disabled = true;
                    select.title = 'Instance : ' + data.instance_name + ' (selection forcee)';
                    return;
                }

                const validIndexes = data.indexes.filter(idx => 
                    !(idx.name === 'documents' && data.indexes.length > 1)
                );
                
                validIndexes.forEach(idx => {
                    const option = document.createElement('option');
                    option.value = idx.name;
                    option.textContent = idx.name;
                    select.appendChild(option);
                });
                
                if (validIndexes.length === 1) {
                    select.value = validIndexes[0].name;
                } else if (data.default) {
                    select.value = data.default;
                }
            } catch (error) {
                console.error('Erreur chargement index:', error);
            }
        }
        
        function toggleSources(sourceId, toggleElement) {
            const sourcesList = document.getElementById(sourceId);
            if (sourcesList) {
                sourcesList.classList.toggle('visible');
                toggleElement.classList.toggle('expanded');
            }
        }
        
        function addMessage(role, content, sources = [], meta = {}) {
            const messages = document.getElementById('messages');
            const welcome = messages.querySelector('.welcome');
            if (welcome) welcome.remove();
            
            messageCounter++;
            const messageId = 'msg-' + messageCounter;
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.id = messageId;
            
            let sourcesHtml = '';
            if (sources.length > 0) {
                const sourceId = 'sources-' + messageCounter;
                sourcesHtml = `
                    <div class="message-sources">
                        <div class="sources-toggle" onclick="toggleSources('${sourceId}', this)">
                            <span class="toggle-icon">▶</span>
                            <span>📎 Sources (${sources.length})</span>
                        </div>
                        <div class="sources-list" id="${sourceId}">
                            ${sources.map(s => `<div class="source-item">• ${escapeHtml(s)}</div>`).join('')}
                        </div>
                    </div>
                `;
            }
            
            // Échapper le contenu pour éviter les problèmes HTML
            const safeContent = formatResponse(content);
            
            let passagesHtml = '';
            if (meta.passages && meta.passages.length > 0) {
                const passId = 'passages-' + messageCounter;
                const items = meta.passages.map(function(p) {
                    let label = escapeHtml(p.file_name || 'Document');
                    if (p.page) label += ' — p.' + p.page;
                    if (p.section) label += ' — ' + escapeHtml(p.section);
                    const txt = escapeHtml((p.text || '').substring(0, 500));
                    return '<div class="passage-item"><div class="passage-meta">📄 ' + label + '</div><div class="passage-text">' + txt + '</div></div>';
                }).join('');
                passagesHtml = `
                    <div class="passages-section">
                        <div class="passages-toggle" onclick="toggleSources('${passId}', this)">
                            <span class="toggle-icon">▶</span>
                            <span>🔍 Passages retrouves (${meta.passages.length})</span>
                        </div>
                        <div class="passages-list" id="${passId}">
                            ${items}
                        </div>
                    </div>
                `;
            }
            
            let metaHtml = '';
            if (meta.processing_time_ms) {
                metaHtml = `<div class="message-meta">⏱️ ${meta.processing_time_ms}ms | 📁 ${meta.index_name || 'N/A'}</div>`;
            }
            
            // Boutons de feedback pour les messages assistant
            let feedbackHtml = '';
            if (role === 'assistant' && meta.query) {
                const feedbackId = 'feedback-' + messageCounter;
                
                // Stocker les données dans un objet JavaScript (pas d'encodage base64)
                feedbackDataStore[feedbackId] = {
                    messageId: messageId,
                    query: meta.query,
                    response: content,
                    sources: sources,
                    indexName: meta.index_name || '',
                    processingTime: meta.processing_time_ms || 0
                };
                
                feedbackHtml = `
                    <div class="feedback-buttons" id="${feedbackId}">
                        <button class="feedback-btn up" onclick="handleFeedback('${feedbackId}', 'up')">
                            👍 Bonne réponse
                        </button>
                        <button class="feedback-btn down" onclick="handleFeedback('${feedbackId}', 'down')">
                            👎 À améliorer
                        </button>
                        <span class="feedback-status"></span>
                    </div>
                `;
            }
            
            div.innerHTML = `
                <div class="message-avatar">${role === 'user' ? '👤' : '<img src="/logo.png" alt="Luciole" class="avatar-logo">'}</div>
                <div class="message-content">
                    <div class="message-text">${safeContent}</div>
                    ${sourcesHtml}
                    ${passagesHtml}
                    ${metaHtml}
                    ${feedbackHtml}
                </div>
            `;
            
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
            
            return messageId;
        }
        
        function addLoading() {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.id = 'loadingMessage';
            div.innerHTML = `
                <div class="message-avatar"><img src="/logo.png" alt="Luciole" class="avatar-logo"></div>
                <div class="message-content">
                    <div class="loading">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                </div>
            `;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function removeLoading() {
            const loading = document.getElementById('loadingMessage');
            if (loading) loading.remove();
        }
        
        async function sendMessage() {
            if (isLoading) return;
            
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            const indexName = document.getElementById('index').value;
            const settings = getSettings();
            
            addMessage('user', message);
            input.value = '';
            conversationHistory.push({ role: 'user', content: message });
            
            isLoading = true;
            document.getElementById('sendBtn').disabled = true;
            addLoading();
            
            try {
                const response = await fetch(`${AGENT_URL}/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: message,
                        index_name: indexName || null,
                        top_k: settings.topK,
                        custom_prompt: settings.customPrompt,
                        deep_search: settings.deepSearch,
                        history: conversationHistory.slice(0, -1)
                    })
                });
                
                const data = await response.json();
                removeLoading();
                
                if (response.ok && data.response) {
                    const sources = (data.sources || []).map(s => 
                        typeof s === 'string' ? s : (s.file_name || JSON.stringify(s))
                    );
                    
                    conversationHistory.push({ role: 'assistant', content: data.response });
                    
                    addMessage('assistant', data.response, sources, {
                        processing_time_ms: data.processing_time_ms,
                        index_name: data.index_name,
                        query: message,
                        passages: data.passages || []
                    });
                } else {
                    addMessage('assistant', `❌ Erreur: ${data.error || data.detail || 'Réponse inattendue'}`);
                }
            } catch (error) {
                removeLoading();
                addMessage('assistant', `❌ Erreur de connexion: ${error.message || error}`);
            }
            
            isLoading = false;
            document.getElementById('sendBtn').disabled = false;
        }
        
        function showNotification(message) {
            const notif = document.createElement('div');
            notif.style.cssText = `
                position: fixed; bottom: 100px; right: 20px;
                background: var(--accent); color: white;
                padding: 0.75rem 1.5rem; border-radius: 8px;
                font-weight: 500; z-index: 2000;
                animation: fadeIn 0.3s ease;
            `;
            notif.textContent = message;
            document.body.appendChild(notif);
            setTimeout(() => notif.remove(), 3000);
        }
        
        // Events
        document.getElementById('messageInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        document.getElementById('messageInput').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
        
        // Fermer modal avec Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
        
        // Badge mail : récupérer le nombre de brouillons en attente
        async function loadMailBadge() {
            try {
                const r = await fetch('/api/mail/health');
                const d = await r.json();
                const badge = document.getElementById('mailDraftsBadge');
                if (badge && d.drafts_pending > 0) {
                    badge.textContent = d.drafts_pending;
                    badge.style.display = 'inline';
                }
            } catch(e) { /* silencieux */ }
        }
        loadMailBadge();

        // Init
        loadIndexes();
        loadSettings();
    </script>
</body>
</html>
"""
    html = html.replace("{{PAGE_TITLE}}", page_title)
    html = html.replace("{{USER_ID}}", user_display)
    html = html.replace("{{USER_ID_RAW}}", user_id or "")
    return html


# ============================================================================
# DASHBOARD PAGE
# ============================================================================

@app.get("/feedbacks", response_class=HTMLResponse)
async def feedbacks_dashboard():
    """Dashboard des feedbacks"""
    service_suffix = f" {SERVICE_NAME}" if SERVICE_NAME else ""
    page_title = f"Luciole Feedbacks{service_suffix}"
    
    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{PAGE_TITLE}} - Dashboard</title>
    <link rel="icon" type="image/png" href="/favicon.png">
    <!-- Fonts: utilise les polices systeme (100% offline) -->
    <style>
        :root {
            --bg-primary: #0B1929;
            --bg-secondary: #0F2237;
            --bg-tertiary: #163050;
            --accent: #FFD76F;
            --accent-dim: #C4952C;
            --accent-glow: rgba(255, 215, 111, 0.25);
            --text-primary: #F8F7F1;
            --text-secondary: #7B96B2;
            --border: #1E3A56;
            --success: #34D399;
            --error: #F87171;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(160deg, #070E18 0%, var(--bg-primary) 40%, #0D1F35 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .header {
            background: rgba(15, 34, 55, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 215, 111, 0.15);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
        }
        
        .logo { display: flex; align-items: center; gap: 0.75rem; }
        .logo h1 {
            font-size: 1.5rem;
            background: linear-gradient(135deg, #FFD76F, #FFF0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header-actions { display: flex; gap: 1rem; }
        
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-primary { background: linear-gradient(135deg, var(--accent), #FFCE60); color: #0B1929; border: none; font-weight: 500; }
        .btn-primary:hover { box-shadow: 0 2px 10px var(--accent-glow); }
        .btn-secondary { background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border); }
        .btn-secondary:hover { border-color: var(--accent); }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--accent);
        }
        
        .stat-value.success { color: var(--success); }
        .stat-value.error { color: var(--error); }
        
        .stat-label { color: var(--text-secondary); margin-top: 0.5rem; }
        
        .section-title {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .table-container {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }
        
        table { width: 100%; border-collapse: collapse; }
        
        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th { background: var(--bg-tertiary); color: var(--text-secondary); font-weight: 500; }
        
        tr:hover { background: var(--bg-tertiary); }
        
        .badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .badge.up { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .badge.down { background: rgba(239, 68, 68, 0.2); color: var(--error); }
        
        .truncate { max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        
        .empty-state { text-align: center; padding: 3rem; color: var(--text-secondary); }
        
        .filters { display: flex; gap: 1rem; margin-bottom: 1rem; }
        .filters select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: inherit;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <h1>📊 {{PAGE_TITLE}}</h1>
        </div>
        <div class="header-actions">
            <a href="/config" class="btn btn-secondary">← Retour a la config</a>
            <a href="/api/feedbacks/export" class="btn btn-primary">📥 Export CSV</a>
        </div>
    </header>
    
    <div class="container">
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <div class="stat-value" id="statTotal">-</div>
                <div class="stat-label">Total feedbacks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success" id="statUp">-</div>
                <div class="stat-label">👍 Positifs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value error" id="statDown">-</div>
                <div class="stat-label">👎 Négatifs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="statRate">-</div>
                <div class="stat-label">Taux de satisfaction</div>
            </div>
        </div>
        
        <h2 class="section-title">📋 Liste des feedbacks</h2>
        
        <div class="filters">
            <select id="filterType" onchange="loadFeedbacks()">
                <option value="">Tous les types</option>
                <option value="up">👍 Positifs</option>
                <option value="down">👎 Négatifs</option>
            </select>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>User</th>
                        <th>Question</th>
                        <th>Feedback</th>
                        <th>Correction</th>
                    </tr>
                </thead>
                <tbody id="feedbacksTable">
                    <tr><td colspan="5" class="empty-state">Chargement...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        async function loadStats() {
            try {
                const response = await fetch('/api/feedbacks/stats');
                const data = await response.json();
                
                document.getElementById('statTotal').textContent = data.total || 0;
                document.getElementById('statUp').textContent = data.up || 0;
                document.getElementById('statDown').textContent = data.down || 0;
                
                const rate = data.total > 0 ? Math.round((data.up / data.total) * 100) : 0;
                document.getElementById('statRate').textContent = rate + '%';
            } catch (e) {
                console.error('Erreur chargement stats:', e);
            }
        }
        
        async function loadFeedbacks() {
            try {
                const filterType = document.getElementById('filterType').value;
                let url = '/api/feedbacks?limit=100';
                if (filterType) url += `&feedback_type=${filterType}`;
                
                const response = await fetch(url);
                const data = await response.json();
                
                const tbody = document.getElementById('feedbacksTable');
                
                if (!data.feedbacks || data.feedbacks.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">Aucun feedback pour le moment</td></tr>';
                    return;
                }
                
                tbody.innerHTML = data.feedbacks.map(fb => `
                    <tr>
                        <td>${new Date(fb.timestamp).toLocaleString('fr-FR')}</td>
                        <td>${fb.user_id || 'Anonyme'}</td>
                        <td class="truncate" title="${fb.query}">${fb.query}</td>
                        <td><span class="badge ${fb.feedback}">${fb.feedback === 'up' ? '👍' : '👎'}</span></td>
                        <td class="truncate" title="${fb.expected_response || ''}">${fb.expected_response || '-'}</td>
                    </tr>
                `).join('');
            } catch (e) {
                console.error('Erreur chargement feedbacks:', e);
            }
        }
        
        loadStats();
        loadFeedbacks();
    </script>
</body>
</html>
"""
    return html.replace("{{PAGE_TITLE}}", page_title)


# ============================================================================
# CONFIG PAGE - Panneau de configuration
# ============================================================================

@app.get("/config", response_class=HTMLResponse)
async def config_page():
    """Page de configuration des paramètres RAG"""
    service_suffix = f" {SERVICE_NAME}" if SERVICE_NAME else ""
    page_title = f"Luciole Config{service_suffix}"

    html = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{PAGE_TITLE}}</title>
    <link rel="icon" type="image/png" href="/favicon.png">
    <!-- Fonts: utilise les polices systeme (100% offline) -->
    <style>
        :root {
            --bg-primary: #0B1929;
            --bg-secondary: #0F2237;
            --bg-tertiary: #163050;
            --accent: #FFD76F;
            --accent-dim: #C4952C;
            --accent-glow: rgba(255, 215, 111, 0.25);
            --text-primary: #F8F7F1;
            --text-secondary: #7B96B2;
            --border: #1E3A56;
            --success: #34D399;
            --error: #F87171;
            --warning: #FFD76F;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(160deg, #070E18 0%, var(--bg-primary) 40%, #0D1F35 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }
        .header {
            background: rgba(15, 34, 55, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 215, 111, 0.15);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
        }
        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, #FFD76F, #FFF0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-actions { display: flex; gap: 0.75rem; align-items: center; }
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-family: inherit;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }
        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }
        .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent), #FFCE60);
            color: #0B1929;
            border: none;
            font-weight: 500;
        }
        .btn-primary:hover { transform: scale(1.02); box-shadow: 0 2px 10px var(--accent-glow); }
        .btn-success {
            background: var(--success);
            color: white;
            border: none;
            font-weight: 500;
        }
        .btn-success:hover { filter: brightness(1.1); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 0;
            border-bottom: 2px solid var(--border);
            margin-bottom: 1.5rem;
        }
        .tab {
            padding: 0.75rem 1.5rem;
            background: none;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.9rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }
        .tab:hover { color: var(--text-primary); }
        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            font-weight: 500;
        }
        .tab-badge {
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            margin-left: 0.4rem;
            vertical-align: middle;
        }
        .badge-rw { background: var(--success); color: white; }
        .badge-ro { background: var(--text-secondary); color: var(--bg-primary); }

        /* Editor */
        .editor-container { display: none; }
        .editor-container.active { display: block; }
        .editor-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .editor-info {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }
        textarea.code-editor {
            width: 100%;
            min-height: 500px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.6;
            resize: vertical;
            tab-size: 2;
        }
        textarea.code-editor:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        textarea.code-editor.readonly {
            background: var(--bg-primary);
            color: var(--text-secondary);
            cursor: default;
        }

        /* Toast notifications */
        .toast-container {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .toast {
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            font-size: 0.85rem;
            animation: toastIn 0.3s ease;
            max-width: 400px;
        }
        .toast-success { background: var(--success); color: white; }
        .toast-error { background: var(--error); color: white; }
        .toast-info { background: var(--accent); color: white; }
        @keyframes toastIn { from { opacity: 0; transform: translateX(100px); } to { opacity: 1; transform: translateX(0); } }

        /* Panneau modèle LLM actif (lecture seule) */
        .models-panel {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }
        .models-panel h3 {
            font-size: 0.95rem;
            color: var(--accent);
            margin-bottom: 0.75rem;
            display: flex; align-items: center; gap: 0.5rem;
        }
        .models-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
            font-size: 0.85rem;
        }
        .models-table th, .models-table td {
            padding: 0.6rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        .models-table th {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 500;
        }
        .models-table tr:hover { background: var(--bg-tertiary); }
        .badge-active {
            background: var(--success);
            color: white;
            font-size: 0.7rem;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }
        .pull-form {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .pull-form input, .pull-form select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
        }
        .pull-form input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
        .pull-form input.model-name { min-width: 220px; }
        .pull-form .param-group { display: flex; align-items: center; gap: 0.25rem; }
        .pull-form .param-group label { font-size: 0.75rem; color: var(--text-secondary); white-space: nowrap; }
        .pull-form .param-group input { width: 80px; }
        .progress-container {
            margin-top: 0.75rem;
            display: none;
        }
        .progress-container.visible { display: block; }
        .progress-bar-bg {
            width: 100%;
            height: 22px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), #FFF0C0);
            border-radius: 6px;
            transition: width 0.3s ease;
            width: 0%;
        }
        .progress-text {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 500;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }
        .progress-status {
            margin-top: 0.35rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        .activate-form {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .activate-form .param-group input {
            width: 80px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.35rem 0.5rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.8rem;
        }
        .activate-form .param-group label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            white-space: nowrap;
        }
        .activate-form .param-group {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        .btn-sm {
            padding: 0.3rem 0.6rem;
            font-size: 0.75rem;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.2s;
        }
        .btn-sm.btn-activate {
            background: var(--accent);
            color: white;
        }
        .btn-sm.btn-activate:hover { filter: brightness(1.15); }
        .btn-sm:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Help section */
        .help-panel {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 1.5rem;
        }
        .help-panel h3 {
            font-size: 0.95rem;
            color: var(--accent);
            margin-bottom: 0.75rem;
        }
        .help-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }
        .help-item {
            background: var(--bg-tertiary);
            border-radius: 6px;
            padding: 0.75rem;
        }
        .help-item strong { color: var(--accent); font-size: 0.85rem; }
        .help-item p { color: var(--text-secondary); font-size: 0.8rem; margin-top: 0.25rem; line-height: 1.4; }

        /* Status bar */
        .status-bar {
            display: flex;
            gap: 1rem;
            align-items: center;
            margin-bottom: 1rem;
            padding: 0.5rem 0.75rem;
            background: var(--bg-secondary);
            border-radius: 8px;
            font-size: 0.8rem;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.ok { background: var(--success); }
        .status-dot.loading { background: var(--warning); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    </style>
</head>
<body>
    <div class="toast-container" id="toastContainer"></div>

    <header class="header">
        <h1>⚙️ {{PAGE_TITLE}}</h1>
        <div class="header-actions">
            <a href="/config" class="btn btn-secondary">← Retour a la config</a>
            <a href="/feedbacks" class="btn btn-secondary">📊 Dashboard</a>
            <button class="btn btn-success" id="btnReload" onclick="reloadConfig()">🔄 Recharger la config</button>
        </div>
    </header>

    <div class="container">
        <div class="status-bar">
            <span><span class="status-dot ok" id="statusDot"></span> Agent</span>
            <span id="statusText" style="color: var(--text-secondary);">Connecté</span>
            <span style="margin-left: auto; color: var(--text-secondary);">
                Les modifications sont appliquées au clic sur "Recharger la config" — pas besoin de redémarrer les containers.
            </span>
        </div>

        <div class="models-panel" id="modelsPanel">
            <h3 id="modelsPanelTitle">🤖 Modèle LLM actif</h3>
            <div id="modelsContent">
                <p style="color:var(--text-secondary);font-size:0.85rem;">Chargement...</p>
            </div>

            <!-- Gestion dynamique (Ollama) — affichée seulement si hot-swap -->
            <div id="ollamaControls" style="display:none;margin-top:1rem;">
                <div class="pull-form">
                    <h4 style="margin:0 0 0.5rem 0;font-size:0.95rem;">➕ Installer un modèle</h4>
                    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">
                        <input type="text" id="searchModelQuery" placeholder="Rechercher (ex: qwen, llama, mistral)…"
                               style="flex:1;min-width:220px;padding:0.5rem;" onkeydown="if(event.key==='Enter')searchModels()">
                        <button class="btn-sm" onclick="searchModels()">🔍 Rechercher</button>
                        <button class="btn-sm" onclick="toggleManualInput()">✏️ Saisie manuelle</button>
                    </div>
                    <div id="manualInputArea" style="display:none;margin-top:0.5rem;">
                        <input type="text" id="manualModelName" placeholder="nom:tag (ex: qwen3:8b)"
                               style="width:60%;padding:0.5rem;">
                        <button class="btn-sm btn-activate" onclick="pullModel(document.getElementById('manualModelName').value)">Télécharger</button>
                    </div>
                    <div id="searchResults" style="margin-top:0.5rem;"></div>
                    <div id="tagSelector" style="display:none;margin-top:0.5rem;"></div>
                    <div id="pullProgress" style="display:none;margin-top:0.75rem;">
                        <div class="progress-container"><div class="progress-bar-fill" id="pullProgressBar" style="width:0%;"></div></div>
                        <p id="pullProgressText" style="font-size:0.8rem;color:var(--text-secondary);margin:0.25rem 0 0 0;"></p>
                    </div>
                </div>
            </div>

            <!-- Message figé — backend TensorRT-LLM -->
            <p id="trtFrozenMsg" style="display:none;font-size:0.8rem;color:var(--text-secondary);margin-top:0.75rem;">
                Le modèle est fixé au lancement du container TensorRT-LLM.
                La gestion dynamique (pull / delete / activation) n'est pas disponible avec TensorRT-LLM.
                Pour changer de modèle, relancez le container avec la nouvelle image/configuration.
            </p>
        </div>

        <div class="tabs" id="tabsContainer">
            <!-- Tabs generated by JS -->
        </div>

        <div id="editorsContainer">
            <!-- Editors generated by JS -->
        </div>

        <div class="help-panel">
            <h3>📖 Guide des réglages</h3>
            <div class="help-grid">
                <div class="help-item">
                    <strong>settings.yaml → temperature</strong>
                    <p>0.1 = très factuel, peu créatif. 0.5 = équilibré. 0.7+ = créatif mais risque d'hallucinations.</p>
                </div>
                <div class="help-item">
                    <strong>settings.yaml → rerank_top_n</strong>
                    <p>Nombre de documents envoyés au LLM. Plus = contexte riche mais plus lent. Défaut: 20.</p>
                </div>
                <div class="help-item">
                    <strong>settings.yaml → max_tokens</strong>
                    <p>Longueur max de la réponse. 2048 = concis, 4096 = détaillé, 8192 = très long.</p>
                </div>
                <div class="help-item">
                    <strong>settings.yaml → bm25_weight / dense_weight</strong>
                    <p>Équilibre entre recherche lexicale (mots exacts) et sémantique (sens). Somme idéale = 1.0.</p>
                </div>
                <div class="help-item">
                    <strong>prompts.yaml → system_prompt</strong>
                    <p>Le "rôle" donné au LLM. Modifier pour changer le ton, les consignes, le format de réponse.</p>
                </div>
                <div class="help-item">
                    <strong>prompts.yaml → rag_prompt</strong>
                    <p>Template qui injecte le contexte + la question. {context} et {query} sont les placeholders.</p>
                </div>
                <div class="help-item">
                    <strong>synonyms.txt</strong>
                    <p>Synonymes métier pour la recherche. Format: "terme1, terme2, terme3" par ligne.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const FILES = [
            { name: 'settings.yaml', label: '⚙️ settings.yaml', editable: true },
            { name: 'prompts.yaml', label: '💬 prompts.yaml', editable: true },
            { name: 'synonyms.txt', label: '📖 synonyms.txt', editable: true },
            { name: 'mail', label: '📧 Mail', editable: false, special: true }
        ];

        let fileContents = {};
        let originalContents = {};
        let activeTab = FILES[0].name;

        // Toast notification
        function showToast(message, type = 'info') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.textContent = message;
            container.appendChild(toast);
            setTimeout(() => toast.remove(), 4000);
        }

        // Tab switching
        function switchTab(filename) {
            activeTab = filename;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab[data-file="${filename}"]`).classList.add('active');
            document.querySelectorAll('.editor-container').forEach(e => e.classList.remove('active'));
            document.getElementById(`editor-${filename.replace('.', '-')}`).classList.add('active');
        }

        // Load file content
        async function loadFile(filename) {
            try {
                const response = await fetch(`/api/config/${filename}`);
                const data = await response.json();
                if (data.error) {
                    showToast(`Erreur: ${data.error}`, 'error');
                    return;
                }
                fileContents[filename] = data.content;
                originalContents[filename] = data.content;
                const editorId = `textarea-${filename.replace('.', '-')}`;
                const textarea = document.getElementById(editorId);
                if (textarea) textarea.value = data.content;
            } catch (e) {
                showToast(`Erreur chargement ${filename}: ${e}`, 'error');
            }
        }

        // Save file
        async function saveFile(filename) {
            const editorId = `textarea-${filename.replace('.', '-')}`;
            const textarea = document.getElementById(editorId);
            const content = textarea.value;

            try {
                const response = await fetch('/api/config/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename, content })
                });
                const data = await response.json();
                if (data.status === 'ok') {
                    showToast(`✅ ${filename} sauvegardé`, 'success');
                    originalContents[filename] = content;
                    updateModifiedIndicator(filename);
                } else {
                    showToast(`❌ ${data.message}`, 'error');
                }
            } catch (e) {
                showToast(`Erreur: ${e}`, 'error');
            }
        }

        // Reload agent config
        async function reloadConfig() {
            const btn = document.getElementById('btnReload');
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');

            btn.disabled = true;
            btn.textContent = '⏳ Rechargement...';
            statusDot.className = 'status-dot loading';
            statusText.textContent = 'Rechargement en cours...';

            // First, save any modified files
            for (const file of FILES) {
                if (!file.editable) continue;
                const editorId = `textarea-${file.name.replace('.', '-')}`;
                const textarea = document.getElementById(editorId);
                if (textarea && textarea.value !== originalContents[file.name]) {
                    await saveFile(file.name);
                }
            }

            try {
                const response = await fetch('/api/config/reload', { method: 'POST' });
                const data = await response.json();
                if (data.status === 'ok') {
                    showToast('🔄 Configuration rechargée ! Les prochaines requêtes utiliseront les nouveaux paramètres.', 'success');
                    statusDot.className = 'status-dot ok';
                    statusText.textContent = 'Config rechargée';
                } else {
                    showToast(`❌ Erreur: ${data.message}`, 'error');
                    statusDot.className = 'status-dot ok';
                    statusText.textContent = 'Erreur rechargement';
                }
            } catch (e) {
                showToast(`❌ Erreur communication: ${e}`, 'error');
                statusDot.className = 'status-dot ok';
                statusText.textContent = 'Erreur';
            }

            btn.disabled = false;
            btn.textContent = '🔄 Recharger la config';
        }

        // Track modifications
        function updateModifiedIndicator(filename) {
            const tab = document.querySelector(`.tab[data-file="${filename}"]`);
            if (!tab) return;
            const dot = tab.querySelector('.modified-dot');
            const editorId = `textarea-${filename.replace('.', '-')}`;
            const textarea = document.getElementById(editorId);
            if (textarea && textarea.value !== originalContents[filename]) {
                if (!dot) {
                    const span = document.createElement('span');
                    span.className = 'modified-dot';
                    span.style.cssText = 'width:6px;height:6px;background:var(--warning);border-radius:50%;display:inline-block;margin-left:6px;';
                    tab.appendChild(span);
                }
            } else {
                if (dot) dot.remove();
            }
        }

        // Handle Tab key in textareas
        function handleTabKey(e) {
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = e.target.selectionStart;
                const end = e.target.selectionEnd;
                e.target.value = e.target.value.substring(0, start) + '  ' + e.target.value.substring(end);
                e.target.selectionStart = e.target.selectionEnd = start + 2;
            }
        }

        // Initialize
        async function init() {
            const tabsContainer = document.getElementById('tabsContainer');
            const editorsContainer = document.getElementById('editorsContainer');

            // Create tabs and editors
            for (const file of FILES) {
                // Tab button
                const tab = document.createElement('button');
                tab.className = `tab${file.name === activeTab ? ' active' : ''}`;
                tab.dataset.file = file.name;
                tab.onclick = () => switchTab(file.name);

                if (file.special) {
                    // Badge santé mail (chargé dynamiquement)
                    tab.innerHTML = `${file.label} <span id="mailTabBadge" style="font-size:0.7rem;padding:0.15rem 0.4rem;border-radius:4px;margin-left:0.3rem;background:var(--text-secondary);color:var(--bg-primary)">…</span>`;
                } else {
                    tab.innerHTML = `${file.label} <span class="tab-badge ${file.editable ? 'badge-rw' : 'badge-ro'}">${file.editable ? 'RW' : 'RO'}</span>`;
                }
                tabsContainer.appendChild(tab);

                const editorId = `editor-${file.name.replace(/\./g, '-')}`;

                // Editor container
                const editorDiv = document.createElement('div');
                editorDiv.id = editorId;
                editorDiv.className = `editor-container${file.name === activeTab ? ' active' : ''}`;

                if (file.special) {
                    // Panneau Mail — formulaire complet
                    editorDiv.innerHTML = getMailPanelHTML();
                    editorsContainer.appendChild(editorDiv);
                } else {
                    const header = document.createElement('div');
                    header.className = 'editor-header';
                    header.innerHTML = `
                        <span class="editor-info">${file.name}</span>
                        <div style="display:flex;gap:0.5rem;">
                            ${file.editable ? `<button class="btn btn-primary" onclick="saveFile('${file.name}')">💾 Sauvegarder</button>` : ''}
                            <button class="btn btn-secondary" onclick="loadFile('${file.name}')">↻ Recharger</button>
                        </div>
                    `;

                    const textarea = document.createElement('textarea');
                    textarea.id = `textarea-${file.name.replace(/\./g, '-')}`;
                    textarea.className = `code-editor${file.editable ? '' : ' readonly'}`;
                    textarea.readOnly = !file.editable;
                    textarea.spellcheck = false;
                    textarea.onkeydown = handleTabKey;
                    if (file.editable) {
                        textarea.oninput = () => updateModifiedIndicator(file.name);
                    }

                    editorDiv.appendChild(header);
                    editorDiv.appendChild(textarea);
                    editorsContainer.appendChild(editorDiv);
                }
            }

            // Load all YAML/config files
            for (const file of FILES) {
                if (!file.special) await loadFile(file.name);
            }

            // Charger les paramètres mail + badge santé
            await mailInit();
        }

        // =========================================================
        // MAIL — Panneau de configuration intégré
        // =========================================================

        function getMailPanelHTML() {
            return `
<style>
.mail-panel { padding: 0.25rem 0; }
.mail-panel .mp-section {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
}
.mail-panel .mp-section h4 {
    color: var(--accent-light); font-size: 0.9rem;
    margin-bottom: 1rem; font-weight: 600;
}
.mail-panel label {
    display: block; color: var(--text-secondary);
    font-size: 0.8rem; margin-top: 0.75rem; margin-bottom: 0.25rem;
}
.mail-panel input, .mail-panel select, .mail-panel textarea {
    width: 100%; background: var(--bg-tertiary); border: 1px solid var(--border);
    border-radius: 7px; color: var(--text-primary); padding: 0.5rem 0.75rem;
    font-family: inherit; font-size: 0.85rem;
}
.mail-panel input:focus, .mail-panel select:focus, .mail-panel textarea:focus {
    outline: none; border-color: var(--accent);
}
.mail-panel .mp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media (max-width: 700px) { .mail-panel .mp-grid { grid-template-columns: 1fr; } }
.mail-panel .mp-toggle-row {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.6rem 0.75rem; background: var(--bg-tertiary);
    border-radius: 8px; margin-bottom: 0.75rem;
}
.mail-panel .mp-toggle-row label { margin: 0; color: var(--text-primary); font-size: 0.85rem; }
.mp-health { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
.mp-health-item { font-size: 0.82rem; display: flex; align-items: center; gap: 0.4rem; }
.mp-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.mp-dot.ok { background: var(--success); }
.mp-dot.err { background: var(--error); }
.mp-dot.off { background: var(--text-secondary); }
.mp-test-result {
    padding: 0.6rem 0.9rem; border-radius: 7px; font-size: 0.82rem;
    margin-top: 0.75rem;
}
.mp-test-result.ok  { background: rgba(16,185,129,.1); border: 1px solid var(--success); color: var(--success); }
.mp-test-result.err { background: rgba(239,68,68,.1); border: 1px solid var(--error); color: var(--error); }
.mp-btn {
    padding: 0.5rem 1.1rem; border-radius: 7px; border: none; font-family: inherit;
    font-size: 0.82rem; cursor: pointer; transition: all 0.2s; font-weight: 500;
}
.mp-btn.primary { background: var(--accent); color: #fff; }
.mp-btn.primary:hover { background: var(--accent-light); }
.mp-btn.secondary {
    background: var(--bg-tertiary); color: var(--text-primary);
    border: 1px solid var(--border);
}
.mp-btn.secondary:hover { border-color: var(--accent); color: var(--accent); }
.mp-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.mp-tests-row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-end; margin-top: 0.5rem; }
.mp-tests-row > div { flex: 1; min-width: 200px; }
.mp-history table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.5rem; }
.mp-history th, .mp-history td { padding: 0.35rem 0.5rem; text-align: left; }
.mp-history thead th { color: var(--text-secondary); }
.mp-history tbody tr { border-top: 1px solid var(--border); }
</style>

<div class="mail-panel">

  <!-- Statut de santé -->
  <div class="mp-section">
    <h4>📊 Statut du module mail</h4>
    <div class="mp-health" id="mpHealth">
      <span style="color:var(--text-secondary);font-size:0.82rem">Chargement…</span>
    </div>
  </div>

  <!-- Paramètres IMAP/SMTP -->
  <div class="mp-section">
    <h4>⚙️ Configuration IMAP / SMTP</h4>
    <div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap">
      <button class="mp-btn secondary" onclick="mailPresetLocal()" title="Pré-remplit avec les valeurs du serveur luciole-mail local">
        🏠 Preset luciole-mail local
      </button>
      <span style="font-size:0.75rem;color:var(--text-secondary)">
        Remplit automatiquement les champs pour le test LAN avec GreenMail
      </span>
    </div>
    <div class="mp-toggle-row">
      <input type="checkbox" id="mpEnabled" style="width:auto;margin:0">
      <label for="mpEnabled">Module mail activé</label>
    </div>
    <div class="mp-toggle-row" style="margin-bottom:1.25rem">
      <input type="checkbox" id="mpAutoReply" style="width:auto;margin:0">
      <label for="mpAutoReply">Auto-réponse activée
        <small style="color:var(--text-secondary);font-size:0.75rem;display:block;margin-top:2px">
          Si activée : Luciole envoie directement sans validation humaine (sauf spam et demande humaine explicite).
          Si désactivée : tout passe par brouillon pour validation.
        </small>
      </label>
    </div>
    <div class="mp-grid">
      <div>
        <strong style="color:var(--accent-light);font-size:0.82rem">📥 Réception (IMAP)</strong>
        <label>Hôte IMAP</label><input id="mpImapHost" placeholder="mail.entreprise.fr">
        <label>Port</label><input id="mpImapPort" type="number" value="993">
        <label>SSL/TLS</label>
        <select id="mpImapSsl">
          <option value="true">Oui (recommandé)</option>
          <option value="false">Non</option>
        </select>
        <label>Utilisateur</label><input id="mpImapUser" placeholder="luciole@entreprise.fr">
        <label>Mot de passe <small style="color:var(--text-secondary)">(vide = conserver)</small></label>
        <input id="mpImapPass" type="password" placeholder="••••••••" autocomplete="new-password">
        <label>Dossier IMAP</label><input id="mpImapFolder" value="INBOX">
        <label>Polling (secondes)</label><input id="mpPoll" type="number" value="60">
      </div>
      <div>
        <strong style="color:var(--accent-light);font-size:0.82rem">📤 Envoi (SMTP)</strong>
        <label>Hôte SMTP</label><input id="mpSmtpHost" placeholder="smtp.entreprise.fr">
        <label>Port</label><input id="mpSmtpPort" type="number" value="465">
        <label>TLS</label>
        <select id="mpSmtpTls">
          <option value="true">Oui (recommandé)</option>
          <option value="false">Non</option>
        </select>
        <label>Utilisateur</label><input id="mpSmtpUser" placeholder="luciole@entreprise.fr">
        <label>Mot de passe <small style="color:var(--text-secondary)">(vide = conserver)</small></label>
        <input id="mpSmtpPass" type="password" placeholder="••••••••" autocomplete="new-password">
        <label>Nom affiché</label><input id="mpFromName" value="Luciole">
        <label>Adresse expéditeur</label><input id="mpFromAddr" placeholder="luciole@entreprise.fr">
      </div>
    </div>
    <label>Signature (optionnelle)</label>
    <textarea id="mpSignature" rows="2" placeholder="--\nLuciole — Assistant documentaire"></textarea>
    <div class="mp-grid" style="margin-top:0">
      <div>
        <label>Seuil confiance (0–1) — sous ce seuil : brouillon</label>
        <input id="mpConf" type="number" min="0" max="1" step="0.05" value="0.75">
      </div>
      <div>
        <label>Seuil risque (0–1) — au-dessus : brouillon / escalade</label>
        <input id="mpRisk" type="number" min="0" max="1" step="0.05" value="0.40">
      </div>
    </div>
    <label>Domaines autorisés (un par ligne — vide = tous acceptés)</label>
    <textarea id="mpAllowed" rows="2" placeholder="entreprise.fr\nfiliale.fr"></textarea>
    <label>Domaines bloqués (un par ligne)</label>
    <textarea id="mpBlocked" rows="2" placeholder="spam.com"></textarea>
    <label>Mots-clés sensibles — forcent le brouillon (un par ligne)</label>
    <textarea id="mpKeywords" rows="3" placeholder="licenciement\ncontentieux\nrgpd"></textarea>
    <label>Index RAG à utiliser</label>
    <input id="mpIndex" value="documents">
    <label>Taille max pièces jointes (Mo)</label>
    <input id="mpMaxMb" type="number" value="25">
    <div style="display:flex;gap:0.75rem;margin-top:1.25rem;flex-wrap:wrap">
      <button class="mp-btn primary" onclick="mailSave()">💾 Sauvegarder</button>
    </div>
    <div id="mpSaveResult"></div>
  </div>

  <!-- Tests de connexion -->
  <div class="mp-section">
    <h4>🧪 Tests de configuration</h4>
    <div class="mp-tests-row">
      <div>
        <button class="mp-btn secondary" id="mpBtnImapTest" onclick="mailTestImap()">📥 Tester IMAP</button>
        <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.3rem">
          Vérifie uniquement la connexion IMAP
        </p>
      </div>
      <div>
        <button class="mp-btn secondary" id="mpBtnSmtpTest" onclick="mailTestSmtp()">📤 Tester SMTP</button>
        <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.3rem">
          Vérifie uniquement la connexion SMTP
        </p>
      </div>
      <div>
        <input id="mpTestRecip" placeholder="admin@entreprise.fr"
          style="background:var(--bg-tertiary);border:1px solid var(--border);border-radius:7px;
                 color:var(--text-primary);padding:0.45rem 0.7rem;font-family:inherit;
                 width:100%;font-size:0.82rem;margin-bottom:0.5rem">
        <button class="mp-btn secondary" id="mpBtnSend" onclick="mailTestSend()">📧 Envoyer un mail de test</button>
        <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.3rem">
          Envoie un vrai email vers l'adresse ci-dessus
        </p>
      </div>
    </div>
    <div id="mpTestResult"></div>
    <div class="mp-history" id="mpTestHistory">
      <p style="color:var(--text-secondary);font-size:0.78rem;margin-top:0.75rem">
        Chargement de l'historique des tests…
      </p>
    </div>
  </div>

</div>`;
        }

        // ── Mail JS functions ──────────────────────────────────────────

        async function mailInit() {
            await mailLoadSettings();
            await mailLoadHealth();
            await mailLoadTestHistory();
        }

        async function mailLoadSettings() {
            try {
                const s = await (await fetch('/api/mail/settings')).json();
                document.getElementById('mpEnabled').checked = s.mail_enabled || false;
                document.getElementById('mpAutoReply').checked = s.auto_reply_enabled || false;
                document.getElementById('mpImapHost').value  = s.imap_host  || '';
                document.getElementById('mpImapPort').value  = s.imap_port  || 993;
                document.getElementById('mpImapSsl').value   = s.imap_use_ssl ? 'true' : 'false';
                document.getElementById('mpImapUser').value  = s.imap_username || '';
                if (s.imap_has_password) document.getElementById('mpImapPass').placeholder = '•••••••• (défini)';
                document.getElementById('mpImapFolder').value = s.imap_folder || 'INBOX';
                document.getElementById('mpPoll').value      = s.imap_poll_interval_seconds || 60;
                document.getElementById('mpSmtpHost').value  = s.smtp_host  || '';
                document.getElementById('mpSmtpPort').value  = s.smtp_port  || 465;
                document.getElementById('mpSmtpTls').value   = s.smtp_use_tls ? 'true' : 'false';
                document.getElementById('mpSmtpUser').value  = s.smtp_username || '';
                if (s.smtp_has_password) document.getElementById('mpSmtpPass').placeholder = '•••••••• (défini)';
                document.getElementById('mpFromName').value  = s.from_name  || 'Luciole';
                document.getElementById('mpFromAddr').value  = s.from_address || '';
                document.getElementById('mpSignature').value = s.signature  || '';
                document.getElementById('mpConf').value      = s.confidence_threshold ?? 0.75;
                document.getElementById('mpRisk').value      = s.risk_threshold ?? 0.40;
                document.getElementById('mpAllowed').value   = (s.allowed_sender_domains  || []).join('\n');
                document.getElementById('mpBlocked').value   = (s.blocked_sender_domains  || []).join('\n');
                document.getElementById('mpKeywords').value  = (s.sensitive_keywords      || []).join('\n');
                document.getElementById('mpIndex').value     = s.index_name || 'documents';
                document.getElementById('mpMaxMb').value     = s.max_attachment_size_mb || 25;
            } catch(e) {
                console.error('Erreur chargement settings mail:', e);
            }
        }

        async function mailLoadHealth() {
            try {
                const h = await (await fetch('/api/mail/health')).json();
                const enabled = h.mail_enabled;
                const configured = h.configured;
                const s = h.stats_24h || {};
                const pending = h.drafts_pending || 0;

                // Badge de l'onglet
                const badge = document.getElementById('mailTabBadge');
                if (badge) {
                    if (!configured) {
                        badge.textContent = 'NON CONFIG';
                        badge.style.background = 'var(--text-secondary)';
                    } else if (!enabled) {
                        badge.textContent = 'DÉSACTIVÉ';
                        badge.style.background = 'var(--warning)';
                        badge.style.color = '#000';
                    } else {
                        badge.textContent = pending > 0 ? `${pending} brouillon${pending>1?'s':''}` : 'ACTIF';
                        badge.style.background = pending > 0 ? 'var(--error)' : 'var(--success)';
                    }
                }

                // Bloc santé
                const el = document.getElementById('mpHealth');
                if (!el) return;
                const lt = h.last_test_connection;
                const imapOk = lt && lt.imap_status === 'ok';
                const smtpOk = lt && lt.smtp_status === 'ok';
                el.innerHTML = `
                    <div class="mp-health-item">
                        <span class="mp-dot ${enabled ? 'ok' : 'off'}"></span>
                        Module ${enabled ? '<b>actif</b>' : 'désactivé'}
                    </div>
                    <div class="mp-health-item">
                        <span class="mp-dot ${!configured ? 'off' : imapOk ? 'ok' : 'err'}"></span>
                        IMAP ${!configured ? '—' : imapOk ? '✓' : (lt ? lt.imap_error_code || '✗' : 'non testé')}
                    </div>
                    <div class="mp-health-item">
                        <span class="mp-dot ${!configured ? 'off' : smtpOk ? 'ok' : 'err'}"></span>
                        SMTP ${!configured ? '—' : smtpOk ? '✓' : (lt ? lt.smtp_error_code || '✗' : 'non testé')}
                    </div>
                    <div class="mp-health-item" style="margin-left:auto;color:var(--text-secondary)">
                        24h — reçus: ${s.received||0} · brouillons: ${s.drafts_created||0}
                        · envoyés: ${s.sent||0} · erreurs: ${s.errors||0}
                        ${pending > 0 ? `· <b style="color:var(--warning)">${pending} en attente</b>` : ''}
                    </div>`;
            } catch(e) {
                const el = document.getElementById('mpHealth');
                if (el) el.innerHTML = '<span style="color:var(--text-secondary);font-size:0.82rem">Module mail non chargé</span>';
            }
        }

        function mpToLines(str) {
            return (str || '').split('\n').map(s => s.trim()).filter(s => s);
        }

        async function mailSave() {
            const payload = {
                mail_enabled:               document.getElementById('mpEnabled').checked,
                auto_reply_enabled:         document.getElementById('mpAutoReply').checked,
                imap_host:                  document.getElementById('mpImapHost').value.trim() || null,
                imap_port:                  parseInt(document.getElementById('mpImapPort').value) || 993,
                imap_use_ssl:               document.getElementById('mpImapSsl').value === 'true',
                imap_username:              document.getElementById('mpImapUser').value.trim() || null,
                imap_password:              document.getElementById('mpImapPass').value || '',
                imap_folder:                document.getElementById('mpImapFolder').value.trim() || 'INBOX',
                imap_poll_interval_seconds: parseInt(document.getElementById('mpPoll').value) || 60,
                smtp_host:                  document.getElementById('mpSmtpHost').value.trim() || null,
                smtp_port:                  parseInt(document.getElementById('mpSmtpPort').value) || 465,
                smtp_use_tls:               document.getElementById('mpSmtpTls').value === 'true',
                smtp_username:              document.getElementById('mpSmtpUser').value.trim() || null,
                smtp_password:              document.getElementById('mpSmtpPass').value || '',
                from_name:                  document.getElementById('mpFromName').value.trim() || 'Luciole',
                from_address:               document.getElementById('mpFromAddr').value.trim() || null,
                signature:                  document.getElementById('mpSignature').value,
                confidence_threshold:       parseFloat(document.getElementById('mpConf').value) || 0.75,
                risk_threshold:             parseFloat(document.getElementById('mpRisk').value) || 0.40,
                allowed_sender_domains:     mpToLines(document.getElementById('mpAllowed').value),
                blocked_sender_domains:     mpToLines(document.getElementById('mpBlocked').value),
                sensitive_keywords:         mpToLines(document.getElementById('mpKeywords').value),
                index_name:                 document.getElementById('mpIndex').value.trim() || 'documents',
                max_attachment_size_mb:     parseInt(document.getElementById('mpMaxMb').value) || 25,
            };
            try {
                const r = await fetch('/api/mail/settings', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const d = await r.json();
                const el = document.getElementById('mpSaveResult');
                if (d.status === 'ok') {
                    showToast('✅ Paramètres mail sauvegardés', 'success');
                    el.innerHTML = '<div class="mp-test-result ok">✅ Sauvegardé</div>';
                    await mailLoadHealth();
                } else {
                    el.innerHTML = `<div class="mp-test-result err">❌ ${d.detail || 'Erreur'}</div>`;
                }
                setTimeout(() => { if(el) el.innerHTML = ''; }, 4000);
            } catch(e) {
                showToast('❌ Erreur sauvegarde mail', 'error');
            }
        }

        async function mailTestConn() {
            // Délégation vers test IMAP + SMTP séquentiels
            await mailTestImap();
            await mailTestSmtp();
        }

        async function mailTestImap() {
            const btn = document.getElementById('mpBtnImapTest');
            const res = document.getElementById('mpTestResult');
            if (btn) { btn.disabled = true; btn.textContent = '⏳ Test IMAP…'; }
            try {
                const d = await (await fetch('/api/mail/test-connection', {method:'POST'})).json();
                const ok = d.imap?.status === 'ok';
                res.innerHTML = `<div class="mp-test-result ${ok ? 'ok' : 'err'}">
                    IMAP : ${ok ? '✅' : '❌'} ${d.imap?.detail || ''} ${d.imap?.latency_ms ? '('+d.imap.latency_ms+'ms)' : ''}
                    ${d.imap?.error_code ? `<br><small>Code : ${d.imap.error_code}</small>` : ''}
                </div>`;
                await mailLoadTestHistory();
                await mailLoadHealth();
            } catch(e) {
                res.innerHTML = `<div class="mp-test-result err">❌ IMAP : ${e.message}</div>`;
            }
            if (btn) { btn.disabled = false; btn.textContent = '📥 Tester IMAP'; }
        }

        async function mailTestSmtp() {
            const btn = document.getElementById('mpBtnSmtpTest');
            const res = document.getElementById('mpTestResult');
            if (btn) { btn.disabled = true; btn.textContent = '⏳ Test SMTP…'; }
            try {
                const d = await (await fetch('/api/mail/test-connection', {method:'POST'})).json();
                const ok = d.smtp?.status === 'ok';
                res.innerHTML = `<div class="mp-test-result ${ok ? 'ok' : 'err'}">
                    SMTP : ${ok ? '✅' : '❌'} ${d.smtp?.detail || ''} ${d.smtp?.latency_ms ? '('+d.smtp.latency_ms+'ms)' : ''}
                    ${d.smtp?.error_code ? `<br><small>Code : ${d.smtp.error_code}</small>` : ''}
                </div>`;
                await mailLoadTestHistory();
                await mailLoadHealth();
            } catch(e) {
                res.innerHTML = `<div class="mp-test-result err">❌ SMTP : ${e.message}</div>`;
            }
            if (btn) { btn.disabled = false; btn.textContent = '📤 Tester SMTP'; }
        }

        function mailPresetLocal() {
            // Préremplissage avec les valeurs luciole-mail (Greenmail LAN)
            // Ports internes Greenmail : SMTP=3025, IMAP=3143 (dans le réseau Docker)
            document.getElementById('mpEnabled').checked   = true;
            document.getElementById('mpImapHost').value    = 'luciole-mail';
            document.getElementById('mpImapPort').value    = '3143';
            document.getElementById('mpImapSsl').value     = 'false';
            document.getElementById('mpImapUser').value    = 'luciole@local.lan';
            document.getElementById('mpImapPass').value    = 'luciole2024';
            document.getElementById('mpImapFolder').value  = 'INBOX';
            document.getElementById('mpPoll').value        = '60';
            document.getElementById('mpSmtpHost').value    = 'luciole-mail';
            document.getElementById('mpSmtpPort').value    = '3025';
            document.getElementById('mpSmtpTls').value     = 'false';
            document.getElementById('mpSmtpUser').value    = 'luciole@local.lan';
            document.getElementById('mpSmtpPass').value    = 'luciole2024';
            document.getElementById('mpFromName').value    = 'Luciole — Assistant documentaire';
            document.getElementById('mpFromAddr').value    = 'luciole@local.lan';
            document.getElementById('mpSignature').value   = '--\nLuciole — Assistant documentaire\nRéponse générée par IA, validée par un humain.';
            document.getElementById('mpConf').value        = '0.75';
            document.getElementById('mpRisk').value        = '0.40';
            document.getElementById('mpIndex').value       = 'documents';
            document.getElementById('mpMaxMb').value       = '25';
            document.getElementById('mpTestRecip').value   = 'testeur@local.lan';
            showToast('🏠 Preset luciole-mail local appliqué — cliquez Sauvegarder pour confirmer', 'info');
        }

        async function mailTestSend() {
            const recipient = (document.getElementById('mpTestRecip').value || '').trim();
            if (!recipient) { showToast('Saisissez une adresse email destinataire', 'error'); return; }
            const btn = document.getElementById('mpBtnSend');
            const res = document.getElementById('mpTestResult');
            btn.disabled = true; btn.textContent = '⏳ Envoi…';
            try {
                const d = await (await fetch('/api/mail/test-send', {
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({recipient})
                })).json();
                const ok = d.status === 'sent';
                res.innerHTML = `<div class="mp-test-result ${ok ? 'ok' : 'err'}">
                    ${ok ? `✅ Email envoyé à ${d.recipient} (${d.latency_ms}ms)` : `❌ Échec : ${d.error}`}
                </div>`;
                await mailLoadTestHistory();
            } catch(e) {
                res.innerHTML = `<div class="mp-test-result err">❌ ${e.message}</div>`;
            }
            btn.disabled = false; btn.textContent = '📤 Envoyer un mail de test';
        }

        async function mailLoadTestHistory() {
            const el = document.getElementById('mpTestHistory');
            if (!el) return;
            try {
                const d = await (await fetch('/api/mail/test-runs?limit=5')).json();
                if (!d.test_runs || d.test_runs.length === 0) {
                    el.innerHTML = '<p style="color:var(--text-secondary);font-size:0.78rem;margin-top:0.5rem">Aucun test effectué</p>';
                    return;
                }
                el.innerHTML = `<table>
                    <thead><tr><th>Type</th><th>Date</th><th>IMAP</th><th>SMTP</th><th>Durée</th></tr></thead>
                    <tbody>${d.test_runs.map(r => `<tr>
                        <td>${r.test_type}</td>
                        <td style="white-space:nowrap">${r.created_at ? new Date(r.created_at).toLocaleString('fr-FR') : '—'}</td>
                        <td>${r.imap_status === 'ok' ? '✅' : r.imap_status === 'skipped' ? '—' : '❌ '+(r.imap_error_code||'')}</td>
                        <td>${r.smtp_status === 'ok' ? '✅' : r.smtp_status === 'skipped' ? '—' : '❌ '+(r.smtp_error_code||'')}</td>
                        <td>${r.total_duration_ms ? r.total_duration_ms+'ms' : '—'}</td>
                    </tr>`).join('')}</tbody>
                </table>`;
            } catch(e) {}
        }

        // ========== MODÈLE LLM ACTIF + gestion dynamique (Ollama) ==========
        // Le backend est détecté côté agent (LLM_URL). Si supports_hot_swap est
        // vrai (Ollama), on affiche le panneau complet de gestion ; sinon
        // (TensorRT-LLM ou autre backend non-Ollama) on montre le modèle en
        // lecture seule + message figé.

        let __supportsHotSwap = false;

        async function loadModels() {
            const container = document.getElementById('modelsContent');
            const controls = document.getElementById('ollamaControls');
            const frozenMsg = document.getElementById('trtFrozenMsg');
            const title = document.getElementById('modelsPanelTitle');
            try {
                const resp = await fetch('/api/llm/model');
                const data = await resp.json();
                if (data.error && !data.model) {
                    container.innerHTML = `<p style="color:var(--error);font-size:0.85rem;">⚠️ ${data.error}</p>`;
                    return;
                }
                __supportsHotSwap = data.supports_hot_swap === true;
                const label = data.backend_label || data.backend || 'LLM';

                if (__supportsHotSwap) {
                    title.textContent = '🤖 Modèles LLM — ' + label;
                    controls.style.display = 'block';
                    frozenMsg.style.display = 'none';
                    await loadOllamaModels();
                } else {
                    title.textContent = '🤖 Modèle LLM actif — ' + label;
                    controls.style.display = 'none';
                    frozenMsg.style.display = 'block';
                    const model = data.model || 'qwen3-30b-a3b-instruct';
                    const url = data.url || 'http://tensorrt-llm:8000';
                    container.innerHTML = `
                        <table class="models-table">
                            <thead><tr><th>Modèle</th><th>Backend</th><th>URL</th><th>Statut</th></tr></thead>
                            <tbody><tr>
                                <td><strong>${model}</strong> <span class="badge-active">ACTIF</span></td>
                                <td style="font-size:0.85rem;">${label}</td>
                                <td style="font-size:0.8rem;color:var(--text-secondary);">${url}</td>
                                <td style="font-size:0.8rem;color:var(--success);">Fixé au démarrage</td>
                            </tr></tbody>
                        </table>`;
                }
            } catch (e) {
                container.innerHTML = `<p style="color:var(--error);font-size:0.85rem;">⚠️ Impossible de contacter l'agent: ${e}</p>`;
            }
        }

        async function loadOllamaModels() {
            const container = document.getElementById('modelsContent');
            try {
                const resp = await fetch('/api/ollama/models');
                const data = await resp.json();
                if (data.error) {
                    container.innerHTML = `<p style="color:var(--error);font-size:0.85rem;">⚠️ ${data.error}</p>`;
                    return;
                }
                const models = data.models || [];
                if (models.length === 0) {
                    container.innerHTML = `<p style="color:var(--text-secondary);font-size:0.85rem;">Aucun modèle installé. Utilisez la recherche ci-dessous pour en télécharger un.</p>`;
                    return;
                }
                container.innerHTML = `
                    <table class="models-table">
                        <thead><tr><th>Modèle</th><th>Taille</th><th>Params</th><th>Quant.</th><th>Statut</th><th>Actions</th></tr></thead>
                        <tbody>${models.map(m => `
                            <tr>
                                <td><strong>${m.name}</strong></td>
                                <td style="font-size:0.85rem;">${m.size_mb ? (m.size_mb/1000).toFixed(1)+' Go' : '—'}</td>
                                <td style="font-size:0.85rem;">${m.parameter_size || '—'}</td>
                                <td style="font-size:0.85rem;">${m.quantization || '—'}</td>
                                <td>${m.active ? '<span class="badge-active">ACTIF</span>' : '<span style="color:var(--text-secondary);font-size:0.8rem;">installé</span>'}</td>
                                <td>
                                    ${m.active ? '' : `<button class="btn-sm btn-activate" onclick="activateModel('${m.name}')">Activer</button>
                                    <button class="btn-sm" onclick="deleteModel('${m.name}')" style="color:var(--error);">Supprimer</button>`}
                                </td>
                            </tr>`).join('')}
                        </tbody>
                    </table>`;
            } catch (e) {
                container.innerHTML = `<p style="color:var(--error);font-size:0.85rem;">⚠️ ${e}</p>`;
            }
        }

        async function activateModel(name) {
            if (!confirm(`Activer le modèle « ${name} » ?`)) return;
            try {
                const resp = await fetch('/api/ollama/activate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: name})
                });
                const data = await resp.json();
                if (!resp.ok) { alert('Erreur: ' + (data.detail || data.error || resp.status)); return; }
                await loadOllamaModels();
            } catch (e) { alert('Erreur: ' + e); }
        }

        async function deleteModel(name) {
            if (!confirm(`Supprimer définitivement « ${name} » ?`)) return;
            try {
                const resp = await fetch('/api/ollama/models', {
                    method: 'DELETE',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: name})
                });
                const data = await resp.json();
                if (!resp.ok) { alert('Erreur: ' + (data.detail || data.error || resp.status)); return; }
                await loadOllamaModels();
            } catch (e) { alert('Erreur: ' + e); }
        }

        function toggleManualInput() {
            const area = document.getElementById('manualInputArea');
            area.style.display = area.style.display === 'none' ? 'block' : 'none';
        }

        async function searchModels() {
            const q = document.getElementById('searchModelQuery').value.trim();
            const results = document.getElementById('searchResults');
            if (!q) { results.innerHTML = ''; return; }
            results.innerHTML = `<p style="font-size:0.85rem;color:var(--text-secondary);">Recherche…</p>`;
            try {
                const resp = await fetch('/api/ollama/search?q=' + encodeURIComponent(q));
                const data = await resp.json();
                if (data.error) { results.innerHTML = `<p style="color:var(--error);font-size:0.85rem;">${data.error}</p>`; return; }
                const models = data.models || [];
                if (models.length === 0) { results.innerHTML = `<p style="font-size:0.85rem;color:var(--text-secondary);">Aucun résultat.</p>`; return; }
                results.innerHTML = models.map(m => `
                    <div style="border:1px solid var(--border);border-radius:6px;padding:0.5rem;margin-bottom:0.4rem;">
                        <strong>${m.name}</strong> <span style="font-size:0.8rem;color:var(--text-secondary);">${m.pulls ? m.pulls+' pulls' : ''}</span>
                        <div style="font-size:0.8rem;color:var(--text-secondary);">${m.description || ''}</div>
                        <div style="margin-top:0.35rem;">
                            ${(m.tags && m.tags.length ? m.tags : ['latest']).map(t =>
                                `<button class="btn-sm" onclick="pullModel('${m.name}:${t}')">${t}</button>`).join(' ')}
                        </div>
                    </div>`).join('');
            } catch (e) { results.innerHTML = `<p style="color:var(--error);font-size:0.85rem;">${e}</p>`; }
        }

        async function pullModel(modelName) {
            const name = (modelName || '').trim();
            if (!name) { alert('Nom de modèle requis'); return; }
            const box = document.getElementById('pullProgress');
            const bar = document.getElementById('pullProgressBar');
            const txt = document.getElementById('pullProgressText');
            box.style.display = 'block'; bar.style.width = '0%'; txt.textContent = 'Démarrage…';
            try {
                const resp = await fetch('/api/ollama/pull', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: name})
                });
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop();
                    for (const line of lines) {
                        const m = line.match(/^data:\s*(.*)$/m);
                        if (!m) continue;
                        try {
                            const p = JSON.parse(m[1]);
                            if (p.error) { txt.textContent = '❌ ' + p.error; return; }
                            bar.style.width = (p.pct || 0) + '%';
                            txt.textContent = `${p.status || ''} ${p.pct || 0}% ` +
                                (p.total_mb ? `(${p.completed_mb}/${p.total_mb} Mo)` : '');
                            if (p.status === 'done') { txt.textContent = '✅ Terminé'; }
                        } catch (e) {}
                    }
                }
                await loadOllamaModels();
            } catch (e) { txt.textContent = '❌ ' + e; }
        }

        init();
        loadModels();
    </script>
</body>
</html>
"""
    return html.replace("{{PAGE_TITLE}}", page_title)


# ============================================================================
# MAIL PAGES — Paramètres, Dashboard, Brouillons, Erreurs
# ============================================================================

_MAIL_CSS = """
    .mail-nav { display:flex; gap:0.5rem; margin-bottom:1.5rem; }
    .mail-nav a {
        padding:0.5rem 1rem; border-radius:8px;
        background:var(--bg-tertiary); border:1px solid var(--border);
        color:var(--text-secondary); text-decoration:none; font-size:0.85rem;
        transition:all 0.2s;
    }
    .mail-nav a:hover, .mail-nav a.active {
        border-color:var(--accent); color:var(--accent);
    }
    .health-bar {
        display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
        gap:1rem; margin-bottom:1.5rem;
    }
    .health-card {
        background:var(--bg-secondary); border:1px solid var(--border);
        border-radius:12px; padding:1.25rem; text-align:center;
    }
    .health-card .hval { font-size:1.8rem; font-weight:700; color:var(--accent); }
    .health-card .hlbl { color:var(--text-secondary); font-size:0.8rem; margin-top:0.25rem; }
    .status-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; }
    .status-dot.ok  { background:var(--success); }
    .status-dot.err { background:var(--error); }
    .status-dot.unk { background:var(--text-secondary); }
    .mail-form label { display:block; color:var(--text-secondary); font-size:0.85rem; margin-bottom:0.3rem; margin-top:0.9rem; }
    .mail-form input, .mail-form select, .mail-form textarea {
        width:100%; background:var(--bg-tertiary); border:1px solid var(--border);
        border-radius:8px; color:var(--text-primary); padding:0.6rem 0.8rem;
        font-family:inherit; font-size:0.9rem;
    }
    .mail-form input:focus, .mail-form select:focus, .mail-form textarea:focus {
        outline:none; border-color:var(--accent);
    }
    .mail-section {
        background:var(--bg-secondary); border:1px solid var(--border);
        border-radius:12px; padding:1.5rem; margin-bottom:1.5rem;
    }
    .mail-section h3 { font-size:1rem; margin-bottom:1rem; color:var(--accent-light); }
    .two-col { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }
    @media (max-width:700px) { .two-col { grid-template-columns:1fr; } }
    .draft-card {
        background:var(--bg-secondary); border:1px solid var(--border);
        border-radius:12px; padding:1.25rem; margin-bottom:1rem;
    }
    .draft-card .from-line { color:var(--text-secondary); font-size:0.85rem; margin-bottom:0.5rem; }
    .draft-card .subject-line { font-weight:600; margin-bottom:0.75rem; }
    .draft-response {
        background:var(--bg-tertiary); border:1px solid var(--border);
        border-radius:8px; padding:1rem; font-size:0.85rem;
        line-height:1.6; white-space:pre-wrap; margin:0.75rem 0;
        max-height:300px; overflow-y:auto;
    }
    .score-badge {
        display:inline-block; padding:0.2rem 0.6rem; border-radius:6px;
        font-size:0.75rem; font-weight:600; margin-right:0.5rem;
    }
    .score-badge.ok  { background:rgba(16,185,129,.2); color:var(--success); }
    .score-badge.warn { background:rgba(245,158,11,.2); color:var(--warning); }
    .score-badge.bad { background:rgba(239,68,68,.2); color:var(--error); }
    .draft-actions { display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:1rem; }
    .btn-approve { background:var(--success); color:#fff; border:none; padding:0.5rem 1.2rem; border-radius:8px; cursor:pointer; font-family:inherit; }
    .btn-reject  { background:var(--error); color:#fff; border:none; padding:0.5rem 1.2rem; border-radius:8px; cursor:pointer; font-family:inherit; }
    .btn-edit    { background:var(--accent-dim); color:#fff; border:none; padding:0.5rem 1.2rem; border-radius:8px; cursor:pointer; font-family:inherit; }
    .sources-list { font-size:0.75rem; color:var(--text-secondary); margin-top:0.5rem; }
    .test-result { padding:0.75rem 1rem; border-radius:8px; margin-top:0.75rem; font-size:0.85rem; }
    .test-result.ok  { background:rgba(16,185,129,.1); border:1px solid var(--success); color:var(--success); }
    .test-result.err { background:rgba(239,68,68,.1); border:1px solid var(--error); color:var(--error); }
"""

_MAIL_HEADER = """
    <header class="header">
        <div class="logo">
            <span class="logo-icon"><img src="/logo.png" alt="Luciole"></span>
            <h1>{{PAGE_TITLE}}</h1>
        </div>
        <div class="header-right">
            <a href="/" class="dashboard-link">💬 Chat</a>
            <a href="/feedbacks" class="dashboard-link">📊 Dashboard</a>
            <a href="/config" class="dashboard-link">⚙️ Config</a>
            <a href="/mail" class="dashboard-link {{MAIL_ACTIVE}}">📧 Mail
                <span id="mailBadge" style="display:none;background:#ef4444;color:#fff;border-radius:10px;padding:1px 5px;font-size:0.7rem;margin-left:3px"></span>
            </a>
        </div>
    </header>
"""

_MAIL_NAV = """
    <div class="mail-nav">
        <a href="/mail" class="{{ACT_DASH}}">📊 Tableau de bord</a>
        <a href="/mail/settings" class="{{ACT_SETTINGS}}">⚙️ Paramètres</a>
        <a href="/mail/drafts" class="{{ACT_DRAFTS}}">⏳ Brouillons</a>
        <a href="/mail/errors" class="{{ACT_ERRORS}}">❌ Erreurs</a>
    </div>
"""

def _mail_page_shell(title: str, body: str, active: str) -> str:
    nav = (
        _MAIL_NAV
        .replace("{{ACT_DASH}}",     "active" if active == "dash" else "")
        .replace("{{ACT_SETTINGS}}", "active" if active == "settings" else "")
        .replace("{{ACT_DRAFTS}}",   "active" if active == "drafts" else "")
        .replace("{{ACT_ERRORS}}",   "active" if active == "errors" else "")
    )
    header = (
        _MAIL_HEADER
        .replace("{{PAGE_TITLE}}", title)
        .replace("{{MAIL_ACTIVE}}", "active")
    )
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="icon" type="image/png" href="/favicon.png">
<style>
:root {{
    --bg-primary:#0a0a0f; --bg-secondary:#12121a; --bg-tertiary:#1a1a25;
    --accent:#8b5cf6; --accent-dim:#6d28d9; --accent-light:#a78bfa;
    --text-primary:#e5e5e5; --text-secondary:#a0a0a0; --border:#2a2a35;
    --success:#10b981; --error:#ef4444; --warning:#f59e0b;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        background:var(--bg-primary); color:var(--text-primary); min-height:100vh; }}
.header {{ background:var(--bg-secondary); border-bottom:1px solid var(--border);
           padding:1rem 2rem; display:flex; align-items:center; justify-content:space-between; }}
.logo {{ display:flex; align-items:center; gap:0.75rem; }}
.logo-icon {{ width:36px; height:36px; }}
.logo-icon img {{ width:100%; height:100%; object-fit:contain; }}
.logo h1 {{ font-size:1.4rem; font-weight:600;
            background:linear-gradient(135deg,var(--accent),var(--accent-light));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.header-right {{ display:flex; align-items:center; gap:0.75rem; }}
.dashboard-link {{ background:var(--bg-tertiary); border:1px solid var(--border);
                   color:var(--text-primary); padding:0.4rem 0.8rem; border-radius:8px;
                   text-decoration:none; font-size:0.82rem; transition:all 0.2s; }}
.dashboard-link:hover, .dashboard-link.active {{ border-color:var(--accent); color:var(--accent); }}
.container {{ max-width:1100px; margin:0 auto; padding:1.5rem 2rem; }}
.btn {{ padding:0.5rem 1.2rem; border-radius:8px; border:none; font-family:inherit;
        font-size:0.85rem; cursor:pointer; transition:all 0.2s; }}
.btn-primary {{ background:var(--accent); color:#fff; }}
.btn-primary:hover {{ background:var(--accent-light); }}
.btn-secondary {{ background:var(--bg-tertiary); color:var(--text-primary);
                  border:1px solid var(--border); }}
.btn-secondary:hover {{ border-color:var(--accent); }}
.btn:disabled {{ opacity:0.5; cursor:not-allowed; }}
{_MAIL_CSS}
</style>
</head>
<body>
{header}
<div class="container">
{nav}
{body}
</div>
<script>
(async () => {{
    try {{
        const r = await fetch('/api/mail/health');
        const d = await r.json();
        const b = document.getElementById('mailBadge');
        if(b && d.drafts_pending > 0) {{ b.textContent = d.drafts_pending; b.style.display='inline'; }}
    }} catch(e) {{}}
}})();
</script>
</body>
</html>"""


# ── /mail — Tableau de bord ──────────────────────────────────────────────────

@app.get("/mail", response_class=HTMLResponse)
async def mail_dashboard():
    body = """
<script>
async function loadDashboard() {
    try {
        const h = await (await fetch('/api/mail/health')).json();
        const s = h.stats_24h || {};
        document.getElementById('statReceived').textContent  = s.received    || 0;
        document.getElementById('statDrafts').textContent    = s.drafts_created || 0;
        document.getElementById('statSent').textContent      = s.sent        || 0;
        document.getElementById('statErrors').textContent    = s.errors      || 0;
        document.getElementById('statQuarantined').textContent = s.quarantined || 0;
        document.getElementById('pendingCount').textContent  = h.drafts_pending || 0;
        document.getElementById('deadLetters').textContent   = h.dead_letters  || 0;
        const enabled = h.mail_enabled;
        const configured = h.configured;
        document.getElementById('modStatus').innerHTML =
            enabled ? '<span class="status-dot ok"></span>Module actif' :
            configured ? '<span class="status-dot warn"></span>Module désactivé' :
            '<span class="status-dot err"></span>Non configuré';
        const lt = h.last_test_connection;
        if(lt) {
            const imapOk = lt.imap_status === 'ok';
            const smtpOk = lt.smtp_status === 'ok';
            document.getElementById('imapStatus').innerHTML =
                `<span class="status-dot ${imapOk?'ok':'err'}"></span>IMAP ${imapOk?'✓':lt.imap_error_code||'✗'}`;
            document.getElementById('smtpStatus').innerHTML =
                `<span class="status-dot ${smtpOk?'ok':'err'}"></span>SMTP ${smtpOk?'✓':lt.smtp_error_code||'✗'}`;
        }
    } catch(e) {
        document.getElementById('modStatus').textContent = 'Erreur chargement';
    }
    // Derniers brouillons
    try {
        const d = await (await fetch('/api/mail/drafts?limit=3')).json();
        const list = document.getElementById('draftsList');
        if (!d.drafts || d.drafts.length === 0) {
            list.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85rem">Aucun brouillon en attente</p>';
        } else {
            list.innerHTML = d.drafts.map(dr => {
                const inp = dr.inbound || {};
                const conf = (dr.confidence_score || 0).toFixed(2);
                return `<div style="padding:0.6rem 0;border-bottom:1px solid var(--border)">
                    <span style="font-size:0.82rem;color:var(--text-secondary)">${inp.from_address||'?'}</span>
                    <span style="margin-left:0.5rem;font-size:0.85rem">${inp.subject||'(sans sujet)'}</span>
                    <span class="score-badge ${conf>=0.75?'ok':'warn'}" style="float:right">conf:${conf}</span>
                </div>`;
            }).join('');
        }
    } catch(e) {}
    // Dernières erreurs
    try {
        const e = await (await fetch('/api/mail/errors?limit=3')).json();
        const list = document.getElementById('errorsList');
        if (!e.errors || e.errors.length === 0) {
            list.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85rem">Aucune erreur récente</p>';
        } else {
            list.innerHTML = e.errors.map(er =>
                `<div style="padding:0.5rem 0;border-bottom:1px solid var(--border);font-size:0.82rem">
                    <span style="color:var(--error)">${er.error_type}</span>
                    <span style="color:var(--text-secondary);margin-left:0.5rem">${(er.error_message||'').substring(0,80)}</span>
                </div>`
            ).join('');
        }
    } catch(e) {}
}
loadDashboard();
</script>
<div class="health-bar">
    <div class="health-card"><div class="hval" id="modStatus" style="font-size:0.9rem;margin-top:0.3rem">…</div><div class="hlbl">Statut module</div></div>
    <div class="health-card"><div id="imapStatus" style="margin-top:0.3rem;font-size:0.85rem">–</div><div class="hlbl">IMAP</div></div>
    <div class="health-card"><div id="smtpStatus" style="margin-top:0.3rem;font-size:0.85rem">–</div><div class="hlbl">SMTP</div></div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1rem;margin-bottom:1.5rem">
    <div class="health-card"><div class="hval" id="statReceived">–</div><div class="hlbl">Reçus 24h</div></div>
    <div class="health-card"><div class="hval" id="statDrafts">–</div><div class="hlbl">Brouillons 24h</div></div>
    <div class="health-card"><div class="hval" id="statSent">–</div><div class="hlbl">Envoyés 24h</div></div>
    <div class="health-card"><div class="hval" id="statQuarantined">–</div><div class="hlbl">Quarantaine 24h</div></div>
    <div class="health-card"><div class="hval" id="statErrors" style="color:var(--error)">–</div><div class="hlbl">Erreurs 24h</div></div>
    <div class="health-card"><div class="hval" id="pendingCount" style="color:var(--warning)">–</div><div class="hlbl">Brouillons en attente</div></div>
    <div class="health-card"><div class="hval" id="deadLetters" style="color:var(--error)">–</div><div class="hlbl">Dead letters</div></div>
</div>
<div class="two-col">
    <div class="mail-section">
        <h3>⏳ Brouillons en attente <a href="/mail/drafts" style="float:right;font-size:0.8rem;color:var(--accent);text-decoration:none">Voir tous →</a></h3>
        <div id="draftsList"><p style="color:var(--text-secondary);font-size:0.85rem">Chargement…</p></div>
    </div>
    <div class="mail-section">
        <h3>❌ Dernières erreurs <a href="/mail/errors" style="float:right;font-size:0.8rem;color:var(--accent);text-decoration:none">Voir toutes →</a></h3>
        <div id="errorsList"><p style="color:var(--text-secondary);font-size:0.85rem">Chargement…</p></div>
    </div>
</div>
"""
    return HTMLResponse(_mail_page_shell("📧 Mail — Tableau de bord", body, "dash"))


# ── /mail/settings ──────────────────────────────────────────────────────────

@app.get("/mail/settings", response_class=HTMLResponse)
async def mail_settings_page():
    body = """
<div class="mail-section">
    <h3>Configuration IMAP/SMTP</h3>
    <div class="mail-form" id="mailForm">
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
            <label style="margin:0;color:var(--text-primary);font-size:0.95rem">Module mail activé</label>
            <label style="position:relative;display:inline-block;width:48px;height:26px">
                <input type="checkbox" id="mailEnabled" style="opacity:0;width:0;height:0">
                <span onclick="document.getElementById('mailEnabled').click()" style="position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:var(--border);border-radius:26px;transition:.3s" id="toggleSlider"></span>
            </label>
        </div>
        <div class="two-col">
            <div>
                <h4 style="color:var(--accent-light);margin-bottom:0.5rem;font-size:0.9rem">📥 Réception (IMAP)</h4>
                <label>Hôte IMAP</label><input id="imapHost" placeholder="mail.entreprise.fr">
                <label>Port</label><input id="imapPort" type="number" value="993">
                <label>SSL/TLS</label>
                <select id="imapSsl"><option value="true">Oui (recommandé)</option><option value="false">Non</option></select>
                <label>Utilisateur</label><input id="imapUser" placeholder="luciole@entreprise.fr">
                <label>Mot de passe <span style="color:var(--text-secondary);font-size:0.75rem">(vide = conserver l'existant)</span></label>
                <input id="imapPass" type="password" placeholder="••••••••" autocomplete="new-password">
                <label>Dossier IMAP</label><input id="imapFolder" value="INBOX">
                <label>Intervalle de polling (secondes)</label><input id="pollInterval" type="number" value="60">
            </div>
            <div>
                <h4 style="color:var(--accent-light);margin-bottom:0.5rem;font-size:0.9rem">📤 Envoi (SMTP)</h4>
                <label>Hôte SMTP</label><input id="smtpHost" placeholder="smtp.entreprise.fr">
                <label>Port</label><input id="smtpPort" type="number" value="465">
                <label>TLS</label>
                <select id="smtpTls"><option value="true">Oui (recommandé)</option><option value="false">Non</option></select>
                <label>Utilisateur</label><input id="smtpUser" placeholder="luciole@entreprise.fr">
                <label>Mot de passe <span style="color:var(--text-secondary);font-size:0.75rem">(vide = conserver)</span></label>
                <input id="smtpPass" type="password" placeholder="••••••••" autocomplete="new-password">
                <label>Nom affiché</label><input id="fromName" value="Luciole">
                <label>Adresse expéditeur</label><input id="fromAddr" placeholder="luciole@entreprise.fr">
            </div>
        </div>
        <label>Signature (optionnelle)</label>
        <textarea id="signature" rows="3" placeholder="--&#10;Luciole — Assistant documentaire"></textarea>
        <div class="two-col" style="margin-top:0">
            <div>
                <label>Seuil de confiance (0-1) — sous ce seuil : brouillon</label>
                <input id="confThreshold" type="number" min="0" max="1" step="0.05" value="0.75">
            </div>
            <div>
                <label>Seuil de risque (0-1) — au-dessus : brouillon/escalade</label>
                <input id="riskThreshold" type="number" min="0" max="1" step="0.05" value="0.40">
            </div>
        </div>
        <label>Domaines autorisés (un par ligne — vide = tous)</label>
        <textarea id="allowedDomains" rows="3" placeholder="entreprise.fr&#10;filiale.fr"></textarea>
        <label>Domaines bloqués (un par ligne)</label>
        <textarea id="blockedDomains" rows="2" placeholder="spam-domain.com"></textarea>
        <label>Mots-clés sensibles (un par ligne — forcent le brouillon)</label>
        <textarea id="sensitiveKw" rows="4" placeholder="licenciement&#10;contentieux&#10;rgpd"></textarea>
        <label>Index RAG à utiliser</label>
        <input id="indexName" value="documents">
        <label>Taille max pièces jointes (Mo)</label>
        <input id="maxAttMb" type="number" value="25">
        <div style="display:flex;gap:1rem;margin-top:1.5rem">
            <button class="btn btn-primary" onclick="saveSettings()">💾 Sauvegarder</button>
        </div>
        <div id="saveResult"></div>
    </div>
</div>
<div class="mail-section">
    <h3>🧪 Tests de configuration</h3>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-end">
        <div style="flex:1;min-width:200px">
            <button class="btn btn-secondary" onclick="testConnection()" id="btnTestConn">🔌 Tester la connexion</button>
            <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.4rem">Vérifie IMAP + SMTP sans envoyer de message</p>
        </div>
        <div style="flex:1;min-width:200px">
            <input id="testRecipient" placeholder="admin@entreprise.fr" style="background:var(--bg-tertiary);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);padding:0.5rem 0.8rem;font-family:inherit;width:100%;margin-bottom:0.5rem">
            <button class="btn btn-secondary" onclick="testSend()" id="btnTestSend">📤 Envoyer un mail de test</button>
            <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.4rem">Envoie un vrai email vers l'adresse ci-dessus</p>
        </div>
    </div>
    <div id="testResult"></div>
    <div style="margin-top:1.5rem">
        <p style="color:var(--text-secondary);font-size:0.82rem;margin-bottom:0.75rem">Derniers tests</p>
        <div id="testHistory"><p style="color:var(--text-secondary);font-size:0.82rem">Chargement…</p></div>
    </div>
</div>
<script>
async function loadSettings() {
    try {
        const s = await (await fetch('/api/mail/settings')).json();
        document.getElementById('mailEnabled').checked = s.mail_enabled;
        document.getElementById('imapHost').value  = s.imap_host || '';
        document.getElementById('imapPort').value  = s.imap_port || 993;
        document.getElementById('imapSsl').value   = s.imap_use_ssl ? 'true' : 'false';
        document.getElementById('imapUser').value  = s.imap_username || '';
        // Password : placeholder si existant
        if (s.imap_has_password) document.getElementById('imapPass').placeholder = '••••••••  (défini)';
        document.getElementById('imapFolder').value  = s.imap_folder || 'INBOX';
        document.getElementById('pollInterval').value = s.imap_poll_interval_seconds || 60;
        document.getElementById('smtpHost').value  = s.smtp_host || '';
        document.getElementById('smtpPort').value  = s.smtp_port || 465;
        document.getElementById('smtpTls').value   = s.smtp_use_tls ? 'true' : 'false';
        document.getElementById('smtpUser').value  = s.smtp_username || '';
        if (s.smtp_has_password) document.getElementById('smtpPass').placeholder = '••••••••  (défini)';
        document.getElementById('fromName').value  = s.from_name || 'Luciole';
        document.getElementById('fromAddr').value  = s.from_address || '';
        document.getElementById('signature').value = s.signature || '';
        document.getElementById('confThreshold').value = s.confidence_threshold || 0.75;
        document.getElementById('riskThreshold').value = s.risk_threshold || 0.40;
        document.getElementById('allowedDomains').value = (s.allowed_sender_domains||[]).join('\\n');
        document.getElementById('blockedDomains').value = (s.blocked_sender_domains||[]).join('\\n');
        document.getElementById('sensitiveKw').value    = (s.sensitive_keywords||[]).join('\\n');
        document.getElementById('indexName').value = s.index_name || 'documents';
        document.getElementById('maxAttMb').value  = s.max_attachment_size_mb || 25;
    } catch(e) { console.error('Erreur chargement settings mail:', e); }
}

function toLines(str) { return str.split('\\n').map(s=>s.trim()).filter(s=>s); }

async function saveSettings() {
    const payload = {
        mail_enabled:               document.getElementById('mailEnabled').checked,
        imap_host:                  document.getElementById('imapHost').value.trim() || null,
        imap_port:                  parseInt(document.getElementById('imapPort').value) || 993,
        imap_use_ssl:               document.getElementById('imapSsl').value === 'true',
        imap_username:              document.getElementById('imapUser').value.trim() || null,
        imap_password:              document.getElementById('imapPass').value || '',
        imap_folder:                document.getElementById('imapFolder').value.trim() || 'INBOX',
        imap_poll_interval_seconds: parseInt(document.getElementById('pollInterval').value) || 60,
        smtp_host:                  document.getElementById('smtpHost').value.trim() || null,
        smtp_port:                  parseInt(document.getElementById('smtpPort').value) || 465,
        smtp_use_tls:               document.getElementById('smtpTls').value === 'true',
        smtp_username:              document.getElementById('smtpUser').value.trim() || null,
        smtp_password:              document.getElementById('smtpPass').value || '',
        from_name:                  document.getElementById('fromName').value.trim() || 'Luciole',
        from_address:               document.getElementById('fromAddr').value.trim() || null,
        signature:                  document.getElementById('signature').value,
        confidence_threshold:       parseFloat(document.getElementById('confThreshold').value) || 0.75,
        risk_threshold:             parseFloat(document.getElementById('riskThreshold').value) || 0.40,
        allowed_sender_domains:     toLines(document.getElementById('allowedDomains').value),
        blocked_sender_domains:     toLines(document.getElementById('blockedDomains').value),
        sensitive_keywords:         toLines(document.getElementById('sensitiveKw').value),
        index_name:                 document.getElementById('indexName').value.trim() || 'documents',
        max_attachment_size_mb:     parseInt(document.getElementById('maxAttMb').value) || 25,
    };
    const res = await fetch('/api/mail/settings', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload)
    });
    const d = await res.json();
    const el = document.getElementById('saveResult');
    if (d.status === 'ok') {
        el.innerHTML = '<div class="test-result ok">✅ Paramètres sauvegardés</div>';
    } else {
        el.innerHTML = `<div class="test-result err">❌ ${d.detail || 'Erreur'}</div>`;
    }
    setTimeout(()=>el.innerHTML='', 4000);
}

async function testConnection() {
    const btn = document.getElementById('btnTestConn');
    btn.disabled = true; btn.textContent = '⏳ Test en cours…';
    const res = document.getElementById('testResult');
    try {
        const d = await (await fetch('/api/mail/test-connection',{method:'POST'})).json();
        const imapOk = d.imap?.status === 'ok';
        const smtpOk = d.smtp?.status === 'ok';
        res.innerHTML = `
            <div class="test-result ${d.overall==='ok'?'ok':'err'}">
                <strong>Connexion : ${d.overall==='ok'?'✅ OK':'⚠️ Partiel/Erreur'}</strong><br>
                IMAP: ${imapOk?'✅':'❌'} ${d.imap?.detail||''} ${d.imap?.latency_ms?`(${d.imap.latency_ms}ms)`:''}<br>
                SMTP: ${smtpOk?'✅':'❌'} ${d.smtp?.detail||''} ${d.smtp?.latency_ms?`(${d.smtp.latency_ms}ms)`:''}
            </div>`;
    } catch(e) {
        res.innerHTML = `<div class="test-result err">❌ Erreur : ${e.message}</div>`;
    }
    btn.disabled = false; btn.textContent = '🔌 Tester la connexion';
    loadTestHistory();
}

async function testSend() {
    const recipient = document.getElementById('testRecipient').value.trim();
    if (!recipient) { alert('Saisissez une adresse email destinataire'); return; }
    const btn = document.getElementById('btnTestSend');
    btn.disabled = true; btn.textContent = '⏳ Envoi…';
    const res = document.getElementById('testResult');
    try {
        const d = await (await fetch('/api/mail/test-send',{
            method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({recipient})
        })).json();
        res.innerHTML = `<div class="test-result ${d.status==='sent'?'ok':'err'}">
            ${d.status==='sent'?'✅':'❌'} ${d.status==='sent'?`Email envoyé à ${d.recipient} (${d.latency_ms}ms)`:`Échec : ${d.error}`}
        </div>`;
    } catch(e) {
        res.innerHTML = `<div class="test-result err">❌ ${e.message}</div>`;
    }
    btn.disabled = false; btn.textContent = '📤 Envoyer un mail de test';
    loadTestHistory();
}

async function loadTestHistory() {
    try {
        const d = await (await fetch('/api/mail/test-runs?limit=5')).json();
        const list = document.getElementById('testHistory');
        if (!d.test_runs || d.test_runs.length === 0) {
            list.innerHTML = '<p style="color:var(--text-secondary);font-size:0.82rem">Aucun test effectué</p>';
            return;
        }
        list.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:0.8rem">
            <tr style="color:var(--text-secondary)"><th style="text-align:left;padding:0.3rem">Type</th><th>Date</th><th>IMAP</th><th>SMTP</th><th>Durée</th></tr>
            ${d.test_runs.map(r => `<tr style="border-top:1px solid var(--border)">
                <td style="padding:0.3rem">${r.test_type}</td>
                <td>${r.created_at ? new Date(r.created_at).toLocaleString('fr-FR') : '-'}</td>
                <td>${r.imap_status === 'ok' ? '✅' : r.imap_status === 'skipped' ? '—' : `❌ ${r.imap_error_code||''}`}</td>
                <td>${r.smtp_status === 'ok' ? '✅' : r.smtp_status === 'skipped' ? '—' : `❌ ${r.smtp_error_code||''}`}</td>
                <td>${r.total_duration_ms ? r.total_duration_ms+'ms' : '—'}</td>
            </tr>`).join('')}
        </table>`;
    } catch(e) {}
}

loadSettings();
loadTestHistory();
</script>
"""
    return HTMLResponse(_mail_page_shell("📧 Mail — Paramètres", body, "settings"))


# ── /mail/drafts ─────────────────────────────────────────────────────────────

@app.get("/mail/drafts", response_class=HTMLResponse)
async def mail_drafts_page():
    body = """
<div id="draftsContainer"><p style="color:var(--text-secondary)">Chargement…</p></div>
<script>
async function loadDrafts() {
    try {
        const d = await (await fetch('/api/mail/drafts?status=pending&limit=50')).json();
        const container = document.getElementById('draftsContainer');
        if (!d.drafts || d.drafts.length === 0) {
            container.innerHTML = '<div class="mail-section"><p style="color:var(--text-secondary);text-align:center;padding:2rem">✅ Aucun brouillon en attente de validation</p></div>';
            return;
        }
        container.innerHTML = `<p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem">${d.drafts.length} brouillon(s) en attente</p>` +
        d.drafts.map(dr => {
            const inp = dr.inbound || {};
            const conf = (dr.confidence_score||0).toFixed(2);
            const risk = (dr.risk_score||0).toFixed(2);
            const confCls = conf >= 0.75 ? 'ok' : conf >= 0.5 ? 'warn' : 'bad';
            const riskCls = risk < 0.4 ? 'ok' : risk < 0.7 ? 'warn' : 'bad';
            const src = (dr.sources||[]).map(s=>s.file_name||s).join(' · ') || '—';
            const createdAt = dr.created_at ? new Date(dr.created_at).toLocaleString('fr-FR') : '—';
            const expires = dr.expires_at ? new Date(dr.expires_at).toLocaleString('fr-FR') : '—';
            return `<div class="draft-card" id="draft-${dr.id}">
                <div class="from-line">De : <strong>${inp.from_address||'?'}</strong> — ${createdAt}</div>
                <div class="subject-line">📧 ${inp.subject || '(sans sujet)'}</div>
                <div>
                    <span class="score-badge ${confCls}">confiance : ${conf}</span>
                    <span class="score-badge ${riskCls}">risque : ${risk}</span>
                    <span style="font-size:0.75rem;color:var(--text-secondary)">${dr.classification||''} — ${dr.decision_reason||''}</span>
                </div>
                <details style="margin-top:0.75rem">
                    <summary style="cursor:pointer;color:var(--text-secondary);font-size:0.82rem">📨 Voir l'email original</summary>
                    <div style="background:var(--bg-primary);border-radius:8px;padding:0.75rem;margin-top:0.5rem;font-size:0.82rem;white-space:pre-wrap;max-height:200px;overflow-y:auto">${escHtml(inp.body_text||'—')}</div>
                </details>
                <p style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.75rem">Réponse proposée :</p>
                <textarea class="draft-response" id="resp-${dr.id}" rows="6">${escHtml(dr.generated_response||'')}</textarea>
                <div class="sources-list">📎 Sources : ${src}</div>
                <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.3rem">Expire le : ${expires}</div>
                <div class="draft-actions">
                    <button class="btn-approve btn" onclick="approveDraft(${dr.id})">✅ Approuver</button>
                    <button class="btn-edit btn" onclick="approveModified(${dr.id})">✏️ Modifier + Approuver</button>
                    <button class="btn-reject btn" onclick="rejectDraft(${dr.id})">❌ Rejeter</button>
                </div>
                <div id="result-${dr.id}" style="margin-top:0.5rem"></div>
            </div>`;
        }).join('');
    } catch(e) {
        document.getElementById('draftsContainer').innerHTML = `<div class="mail-section" style="color:var(--error)">Erreur : ${e.message}</div>`;
    }
}

function escHtml(t) {
    return String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function approveDraft(id) {
    const res = await fetch(`/api/mail/drafts/${id}/approve`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({modified_response: null})
    });
    const d = await res.json();
    showResult(id, d.status === 'approved', d);
    if(d.status === 'approved') setTimeout(()=>document.getElementById('draft-'+id)?.remove(), 1500);
}

async function approveModified(id) {
    const modified = document.getElementById('resp-'+id)?.value;
    if (!modified?.trim()) { alert('La réponse ne peut pas être vide'); return; }
    const res = await fetch(`/api/mail/drafts/${id}/approve`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({modified_response: modified})
    });
    const d = await res.json();
    showResult(id, d.status === 'approved' || d.status === 'modified_approved', d);
    if(d.status) setTimeout(()=>document.getElementById('draft-'+id)?.remove(), 1500);
}

async function rejectDraft(id) {
    const comment = prompt('Raison du rejet (optionnel) :') || '';
    const res = await fetch(`/api/mail/drafts/${id}/reject`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({comment})
    });
    const d = await res.json();
    showResult(id, d.status === 'rejected', d);
    if(d.status === 'rejected') setTimeout(()=>document.getElementById('draft-'+id)?.remove(), 1500);
}

function showResult(id, ok, d) {
    const el = document.getElementById('result-'+id);
    if (!el) return;
    el.innerHTML = `<div class="test-result ${ok?'ok':'err'}">${ok?'✅':'❌'} ${JSON.stringify(d)}</div>`;
}

loadDrafts();
</script>
"""
    return HTMLResponse(_mail_page_shell("📧 Mail — Brouillons", body, "drafts"))


# ── /mail/errors ─────────────────────────────────────────────────────────────

@app.get("/mail/errors", response_class=HTMLResponse)
async def mail_errors_page():
    body = """
<div class="mail-section">
    <h3>❌ Dead-letters et erreurs actives</h3>
    <div id="errorsContainer"><p style="color:var(--text-secondary)">Chargement…</p></div>
</div>
<script>
async function loadErrors() {
    try {
        const d = await (await fetch('/api/mail/errors?limit=50')).json();
        const container = document.getElementById('errorsContainer');
        if (!d.errors || d.errors.length === 0) {
            container.innerHTML = '<p style="color:var(--success)">✅ Aucune erreur active</p>';
            return;
        }
        container.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:0.82rem">
            <thead><tr style="color:var(--text-secondary)">
                <th style="text-align:left;padding:0.5rem">Date</th>
                <th>Type</th><th>Message</th><th>Retry</th><th>Statut</th><th>Actions</th>
            </tr></thead>
            <tbody>
            ${d.errors.map(e => `<tr style="border-top:1px solid var(--border)" id="err-${e.id}">
                <td style="padding:0.4rem;white-space:nowrap">${e.created_at ? new Date(e.created_at).toLocaleString('fr-FR') : '—'}</td>
                <td><span style="color:var(--warning)">${e.error_type}</span></td>
                <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${e.error_message||''}">${(e.error_message||'').substring(0,80)}</td>
                <td>${e.retry_count||0}/${e.max_retries||3}</td>
                <td><span style="color:${e.status==='exhausted'?'var(--error)':'var(--warning)'}">${e.status}</span></td>
                <td style="white-space:nowrap">
                    ${e.status !== 'exhausted' ? `<button onclick="retryError(${e.id})" style="background:var(--accent-dim);color:#fff;border:none;padding:0.2rem 0.6rem;border-radius:6px;cursor:pointer;font-size:0.75rem">Retenter</button>` : ''}
                    <button onclick="ignoreError(${e.id})" style="background:var(--bg-tertiary);color:var(--text-secondary);border:1px solid var(--border);padding:0.2rem 0.6rem;border-radius:6px;cursor:pointer;font-size:0.75rem;margin-left:0.3rem">Ignorer</button>
                </td>
            </tr>`).join('')}
            </tbody>
        </table>`;
    } catch(e) {
        document.getElementById('errorsContainer').innerHTML = `<p style="color:var(--error)">Erreur : ${e.message}</p>`;
    }
}

async function retryError(id) {
    const r = await fetch(`/api/mail/errors/${id}/retry`, {method:'POST'});
    const d = await r.json();
    if(d.status === 'queued') {
        document.getElementById('err-'+id)?.remove();
        const el = document.createElement('div');
        el.className = 'test-result ok'; el.textContent = '✅ Remis en file d\'attente';
        document.getElementById('errorsContainer').prepend(el);
        setTimeout(()=>el.remove(), 3000);
    }
}

async function ignoreError(id) {
    await fetch(`/api/mail/errors/${id}/ignore`, {method:'POST'});
    document.getElementById('err-'+id)?.remove();
}

loadErrors();
</script>
"""
    return HTMLResponse(_mail_page_shell("📧 Mail — Erreurs", body, "errors"))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FEEDBACK_PORT", 8501))
    uvicorn.run(app, host="0.0.0.0", port=port)
