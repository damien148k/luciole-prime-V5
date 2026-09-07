"""
Interface d'Administration - Ingestion de Documents
UI pour parcourir les fichiers et lancer l'ingestion avec logs en temps réel
V3 : Authentification cookie via config/auth.yaml
"""

import os
import asyncio
from pathlib import Path
from typing import List
from datetime import datetime
import json
import yaml
import sqlite3

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from loguru import logger
from src.ui.auth import (
    verify_credentials, make_session_token, validate_session_token,
    get_login_html, AUTH_COOKIE_NAME,
)

app = FastAPI(title="Luciole RAG - Ingestion", version="3.0")

# Routes publiques (pas d'auth requise)
_PUBLIC_PATHS = {"/login", "/logout", "/logo.png", "/logo.svg", "/favicon.ico", "/favicon.png", "/health"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Vérifie le cookie de session sur toutes les routes sauf publiques."""
    if request.url.path in _PUBLIC_PATHS or request.url.path.startswith("/ws"):
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

# Chemins vers le logo et favicon (relatif au fichier courant)
STATIC_DIR = Path(__file__).parent / "static"
PICS_DIR = Path(__file__).parent.parent.parent / "pics"
LOGO_PATH = str(STATIC_DIR / "logo.png")
FAVICON_PATH = str(STATIC_DIR / "favicon.png")

# Configuration des services (supporte Docker et local)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

# OpenSearch - supporter OPENSEARCH_URL ou OPENSEARCH_HOST/PORT
_opensearch_url = os.environ.get("OPENSEARCH_URL", "")
if _opensearch_url:
    # Parser l'URL (ex: http://luciole-opensearch:9200)
    from urllib.parse import urlparse
    _parsed = urlparse(_opensearch_url)
    OPENSEARCH_HOST = _parsed.hostname or "localhost"
    OPENSEARCH_PORT = _parsed.port or 9200
else:
    OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))

# Configuration
ALLOWED_EXTENSIONS = [
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", 
    ".msg", ".eml", ".txt", ".md", ".rst", ".csv",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"  # Images avec OCR
]

# WebSocket connections pour les logs en temps réel
active_connections: List[WebSocket] = []


# ============================================================================
# WebSocket pour les logs en temps réel
# ============================================================================

async def broadcast_log(message: str, level: str = "info", data: dict = None):
    """Envoie un log à tous les clients connectés"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "data": data or {}
    }
    
    for connection in active_connections:
        try:
            await connection.send_json(log_entry)
        except:
            pass


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint pour les logs en temps réel"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        await websocket.send_json({
            "timestamp": datetime.now().isoformat(),
            "level": "info",
            "message": "✓ Connecté au serveur de logs",
            "data": {}
        })
        
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


# ============================================================================
# Watcher — supervision de l'ingestion
# ============================================================================
# En V3 cette interface LANCAIT l'ingestion (selection d'un dossier, bouton
# « Lancer »). En V4 le watcher est la seule voie d'entree des documents :
# deposer, modifier ou retirer un fichier dans data/ suffit. L'admin ne
# commande donc plus l'ingestion, elle la SUPERVISE — et peut demander une
# reconciliation quand un depot n'a pas ete vu.

WATCHER_URL = os.environ.get("WATCHER_URL", "http://watcher:8090")
# V4 : plus de conteneur agent, le coeur RAG repond directement.
RAG_URL = os.environ.get("RAG_URL") or os.environ.get("AGENT_URL", "http://rag:8000")


async def _watcher(chemin: str, methode: str = "GET", **kw):
    """Appelle l'API du watcher, en rendant l'echec lisible a l'interface."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.request(methode, f"{WATCHER_URL}{chemin}", **kw)
        if r.status_code >= 400:
            return {"error": f"HTTP {r.status_code}", "detail": r.text[:300]}
        return r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"watcher injoignable ({e})")
        return {"error": "watcher injoignable", "detail": str(e)}


@app.get("/api/watcher/status")
async def watcher_status():
    """Etat du service, file d'attente et compte des documents."""
    return await _watcher("/api/watcher/status")


@app.get("/api/watcher/documents")
async def watcher_documents(limit: int = 100, offset: int = 0):
    return await _watcher("/api/watcher/documents",
                          params={"limit": limit, "offset": offset})


@app.get("/api/watcher/jobs")
async def watcher_jobs(limit: int = 50, status: str = ""):
    params = {"limit": limit}
    if status:
        params["status"] = status
    return await _watcher("/api/watcher/jobs", params=params)


@app.get("/api/watcher/errors")
async def watcher_errors(limit: int = 50):
    return await _watcher("/api/watcher/errors", params={"limit": limit})


@app.get("/api/watcher/metrics")
async def watcher_metrics():
    return await _watcher("/api/watcher/metrics")


@app.post("/api/watcher/rescan")
async def watcher_rescan():
    """Reconciliation manuelle : compare le disque et l'index, puis
    enfile les ajouts, les modifications et les suppressions."""
    resultat = await _watcher("/api/watcher/rescan", "POST")
    await broadcast_log("Reconciliation demandee au watcher", "info", resultat)
    return resultat


@app.get("/api/watcher/logs")
async def watcher_logs(lines: int = 200):
    """Dernieres lignes du journal du watcher.

    Le watcher duplique son journal dans backups/watcher/watcher.log (voir
    src/watcher/main.py). L'admin monte ce dossier en lecture seule : elle
    voit ce que fait l'ingestion sans qu'on lui donne acces au demon
    Docker.
    """
    chemin = os.environ.get("WATCHER_LOG_PATH", "/app/backups/watcher/watcher.log")
    if not os.path.exists(chemin):
        return {"error": "journal absent",
                "detail": f"{chemin} n'existe pas encore — le watcher a-t-il demarre ?",
                "lines": []}
    try:
        # Lecture de la fin seule : le fichier peut atteindre 10 Mo.
        with open(chemin, "rb") as f:
            taille = f.seek(0, os.SEEK_END)
            f.seek(max(0, taille - 256 * 1024))
            brut = f.read().decode("utf-8", errors="replace")
        return {"lines": [l for l in brut.splitlines() if l.strip()][-lines:]}
    except Exception as e:  # noqa: BLE001
        return {"error": "journal illisible", "detail": str(e), "lines": []}

@app.get("/api/stats")
async def get_stats():
    """Compte des vecteurs et des documents, plus les reglages affiches.

    La V3 instanciait IngestionPipeline, ce qui chargeait bge-m3 : deux
    gigaoctets de VRAM dans un conteneur d'interface, juste pour compter.
    On interroge desormais Qdrant et OpenSearch en HTTP, et on lit les
    reglages dans settings.yaml.
    """
    import httpx
    reglages, nom = {}, os.environ.get("INSTANCE_NAME", "")
    try:
        with open(_find_settings_yaml(), encoding="utf-8") as f:
            reglages = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"settings.yaml illisible ({e})")
    collection = (reglages.get("qdrant") or {}).get("collection_name") or nom
    index = (reglages.get("opensearch") or {}).get("index_name") or nom

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    opensearch_url = os.environ.get("OPENSEARCH_URL", "http://opensearch:9200")
    vecteurs, documents, erreurs = 0, 0, []
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(f"{qdrant_url}/collections/{collection}")
            vecteurs = ((r.json().get("result") or {}).get("points_count")) or 0
        except Exception as e:  # noqa: BLE001
            erreurs.append(f"qdrant: {e}")
        try:
            r = await client.get(f"{opensearch_url}/{index}/_count")
            documents = r.json().get("count", 0)
        except Exception as e:  # noqa: BLE001
            erreurs.append(f"opensearch: {e}")

    sortie = {
        "qdrant_vectors": vecteurs,
        "opensearch_documents": documents,
        "embedding_model": (reglages.get("embedding") or {}).get("model", ""),
        "chunk_size": (reglages.get("chunking") or {}).get("chunk_size", 0),
    }
    if erreurs:
        sortie["error"] = " ; ".join(erreurs)
    return sortie


@app.get("/api/config")
async def get_config():
    """Retourne la configuration depuis settings.yaml (lecture seule)"""
    try:
        import yaml
        config_path = "config/settings.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Retourner seulement les sections pertinentes
        return {
            "chunking": config.get("chunking", {}),
            "retrieval": config.get("retrieval", {}),
            "embedding": config.get("embedding", {}),
            "reranker": config.get("reranker", {}),
            "config_file": config_path
        }
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/index/{index_name}")
async def delete_index(index_name: str):
    """Supprime un index (Qdrant collection et/ou OpenSearch index)"""
    results = {"qdrant": None, "opensearch": None}
    
    try:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(url=QDRANT_URL, timeout=10)
        
        # Vérifier si la collection existe
        collections = qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if index_name in collection_names:
            qdrant.delete_collection(index_name)
            results["qdrant"] = f"Collection '{index_name}' supprimée"
            await broadcast_log(f"🗑️ Qdrant: collection '{index_name}' supprimée", "success")
        else:
            results["qdrant"] = f"Collection '{index_name}' non trouvée"
            
    except Exception as e:
        results["qdrant"] = f"Erreur: {str(e)}"
        await broadcast_log(f"❌ Erreur Qdrant: {str(e)}", "error")
    
    try:
        from opensearchpy import OpenSearch
        opensearch = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=("admin", "admin"),
            use_ssl=False,
            verify_certs=False
        )
        
        # OpenSearch utilise des noms en minuscules - essayer les deux versions
        os_index_lower = index_name.lower()
        deleted_any = False
        
        # Essayer le nom exact
        if opensearch.indices.exists(index=index_name):
            opensearch.indices.delete(index=index_name)
            results["opensearch"] = f"Index '{index_name}' supprimé"
            await broadcast_log(f"🗑️ OpenSearch: index '{index_name}' supprimé", "success")
            deleted_any = True
        
        # Essayer aussi la version lowercase si différente
        if os_index_lower != index_name and opensearch.indices.exists(index=os_index_lower):
            opensearch.indices.delete(index=os_index_lower)
            if deleted_any:
                results["opensearch"] += f" + '{os_index_lower}' supprimé"
            else:
                results["opensearch"] = f"Index '{os_index_lower}' supprimé"
            await broadcast_log(f"🗑️ OpenSearch: index '{os_index_lower}' supprimé", "success")
            deleted_any = True
        
        if not deleted_any:
            results["opensearch"] = f"Index '{index_name}' non trouvé"
            
    except Exception as e:
        results["opensearch"] = f"Erreur: {str(e)}"
        await broadcast_log(f"❌ Erreur OpenSearch: {str(e)}", "error")
    
    return results


@app.get("/api/indexes")
async def list_indexes():
    """
    Liste tous les index disponibles.
    Fusionne les collections Qdrant et index OpenSearch qui ont le même nom (case-insensitive).
    Qdrant garde la casse originale, OpenSearch est toujours lowercase.
    """
    # Dictionnaire unifié: clé = nom lowercase, valeur = info combinée
    unified_indexes = {}
    qdrant_collections = []
    opensearch_indexes = []
    errors = {}
    
    # 1. Récupérer les collections Qdrant
    try:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(url=QDRANT_URL, timeout=10)
        collections = qdrant.get_collections().collections
        
        for c in collections:
            info = qdrant.get_collection(c.name)
            # qdrant-client 1.19 (V4) ne porte plus vectors_count sur
            # CollectionInfo ; points_count est le compte de reference,
            # et vaut le nombre de vecteurs dans un schema a vecteur
            # unique — celui de Luciole.
            points = getattr(info, "points_count", None) or 0
            vecteurs = getattr(info, "vectors_count", None)
            qdrant_collections.append({
                "name": c.name,
                "vectors_count": points if vecteurs is None else vecteurs,
                "points_count": points,
            })
            
            # Ajouter à l'index unifié (clé = lowercase)
            key = c.name.lower()
            if key not in unified_indexes:
                unified_indexes[key] = {
                    "name": c.name,  # Garder le nom Qdrant (casse originale)
                    "qdrant_vectors": points,
                    "opensearch_docs": 0
                }
            else:
                unified_indexes[key]["qdrant_vectors"] = points
                
    except Exception as e:
        errors["qdrant_error"] = str(e)
    
    # 2. Récupérer les index OpenSearch
    try:
        from opensearchpy import OpenSearch
        opensearch = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=("admin", "admin"),
            use_ssl=False,
            verify_certs=False
        )
        
        indices = opensearch.indices.get_alias(index="*")
        for name, info in indices.items():
            if not name.startswith("."):  # Ignorer les index système
                stats = opensearch.indices.stats(index=name)
                doc_count = stats["indices"][name]["total"]["docs"]["count"]
                opensearch_indexes.append({
                    "name": name,
                    "documents_count": doc_count
                })
                
                # Fusionner avec l'index unifié
                key = name.lower()  # OpenSearch est déjà lowercase
                if key not in unified_indexes:
                    unified_indexes[key] = {
                        "name": name,
                        "qdrant_vectors": 0,
                        "opensearch_docs": doc_count
                    }
                else:
                    unified_indexes[key]["opensearch_docs"] = doc_count
                    
    except Exception as e:
        errors["opensearch_error"] = str(e)
    
    # 3. Construire la liste finale (sans doublons)
    unified_list = list(unified_indexes.values())
    
    return {
        "qdrant_collections": qdrant_collections,
        "opensearch_indexes": opensearch_indexes,
        "unified_indexes": unified_list,  # Liste fusionnée pour l'UI
        **errors
    }


@app.delete("/api/clear-all")
async def clear_all_indexes():
    """Supprime tous les index RAG (documents_dense et documents_bm25)"""
    await broadcast_log("🗑️ Suppression de tous les index RAG...", "warning")
    
    result1 = await delete_index("documents_dense")
    result2 = await delete_index("documents_bm25")
    
    await broadcast_log("✓ Index RAG supprimés. Prêt pour une nouvelle ingestion.", "success")
    
    return {"documents_dense": result1, "documents_bm25": result2}


@app.post("/api/index/{index_name}/export")
async def export_index(index_name: str):
    """
    Exporte un index (Qdrant + OpenSearch) vers des fichiers JSON.
    Les fichiers sont sauvegardés dans /app/backups/ (Docker) ou C:/RAG/backups/ (Windows)
    """
    import json
    from datetime import datetime
    import platform
    
    # Déterminer le chemin de backup selon l'environnement
    if platform.system() == "Windows":
        backup_dir = Path("C:/RAG/backups")
    else:
        # Dans Docker (Linux), utiliser un chemin relatif à l'app
        backup_dir = Path("/app/backups")
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    await broadcast_log(f"📁 Dossier backup: {backup_dir}", "info")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"qdrant": None, "opensearch": None, "backup_path": str(backup_dir)}
    
    await broadcast_log(f"📦 Export de l'index '{index_name}'...", "info")
    
    # Export Qdrant
    try:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
        
        # Récupérer tous les points
        points = []
        offset = None
        while True:
            result = qdrant.scroll(
                collection_name=index_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            points.extend([{
                "id": str(p.id),
                "vector": p.vector,
                "payload": p.payload
            } for p in result[0]])
            offset = result[1]
            if offset is None:
                break
        
        qdrant_file = backup_dir / f"{index_name}_qdrant_{timestamp}.json"
        with open(qdrant_file, "w", encoding="utf-8") as f:
            json.dump({"collection": index_name, "points": points}, f, ensure_ascii=False)
        
        results["qdrant"] = f"Exporté {len(points)} vecteurs → {qdrant_file.name}"
        await broadcast_log(f"✓ Qdrant: {len(points)} vecteurs exportés", "success")
        
    except Exception as e:
        results["qdrant"] = f"Erreur: {str(e)}"
        await broadcast_log(f"❌ Qdrant export error: {e}", "error")
    
    # Export OpenSearch
    try:
        from opensearchpy import OpenSearch
        opensearch = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=("admin", "admin"),
            use_ssl=False,
            verify_certs=False
        )
        
        os_index = index_name.lower()
        if opensearch.indices.exists(index=os_index):
            # Récupérer tous les documents
            docs = []
            result = opensearch.search(
                index=os_index,
                body={"query": {"match_all": {}}, "size": 10000}
            )
            for hit in result["hits"]["hits"]:
                docs.append({
                    "id": hit["_id"],
                    "source": hit["_source"]
                })
            
            os_file = backup_dir / f"{os_index}_opensearch_{timestamp}.json"
            with open(os_file, "w", encoding="utf-8") as f:
                json.dump({"index": os_index, "documents": docs}, f, ensure_ascii=False)
            
            results["opensearch"] = f"Exporté {len(docs)} documents → {os_file.name}"
            await broadcast_log(f"✓ OpenSearch: {len(docs)} documents exportés", "success")
        else:
            results["opensearch"] = "Index non trouvé"
            
    except Exception as e:
        results["opensearch"] = f"Erreur: {str(e)}"
        await broadcast_log(f"❌ OpenSearch export error: {e}", "error")
    
    await broadcast_log(f"📦 Export terminé → {backup_dir}", "success")
    return results


class ImportRequest(BaseModel):
    backup_file: str


@app.post("/api/index/import")
async def import_index(request: ImportRequest):
    """
    Importe un index depuis un fichier JSON de backup.
    Le fichier doit être dans le dossier de backups
    """
    import json
    
    backup_path = Path(request.backup_file)
    if not backup_path.exists():
        backup_path = get_backup_dir() / request.backup_file
    
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail=f"Fichier non trouvé: {backup_path}")
    
    await broadcast_log(f"📥 Import depuis '{backup_path.name}'...", "info")
    
    with open(backup_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Détecter le type (qdrant ou opensearch)
    if "points" in data:
        # Import Qdrant
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance, PointStruct
            
            qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
            collection_name = data["collection"]
            
            # Vérifier si la collection existe (compatible Qdrant v1.7.x)
            existing_collections = [c.name for c in qdrant.get_collections().collections]
            collection_exists = collection_name in existing_collections
            
            # Créer la collection si elle n'existe pas
            if not collection_exists:
                vector_size = len(data["points"][0]["vector"]) if data["points"] else 1024
                await broadcast_log(f"📦 Création de la collection '{collection_name}' (vecteur: {vector_size}d)...", "info")
                qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
            
            # Fonction pour convertir les IDs (string numérique -> int, sinon garder comme UUID)
            def convert_id(id_val):
                if isinstance(id_val, int):
                    return id_val
                if isinstance(id_val, str):
                    # Essayer de convertir en int si c'est un nombre
                    try:
                        return int(id_val)
                    except ValueError:
                        # C'est probablement un UUID, le garder tel quel
                        return id_val
                return id_val
            
            # Insérer les points par batch
            points = [
                PointStruct(
                    id=convert_id(p["id"]),
                    vector=p["vector"],
                    payload=p["payload"]
                ) for p in data["points"]
            ]
            
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                qdrant.upsert(collection_name=collection_name, points=batch)
            
            await broadcast_log(f"✓ Qdrant: {len(points)} vecteurs importés dans '{collection_name}'", "success")
            return {"status": "success", "type": "qdrant", "count": len(points)}
            
        except Exception as e:
            await broadcast_log(f"❌ Qdrant import error: {e}", "error")
            raise HTTPException(status_code=500, detail=str(e))
    
    elif "documents" in data:
        # Import OpenSearch
        try:
            from opensearchpy import OpenSearch
            
            opensearch = OpenSearch(
                hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
                http_auth=("admin", "admin"),
                use_ssl=False,
                verify_certs=False
            )
            
            index_name = data["index"]
            
            # Créer l'index s'il n'existe pas
            if not opensearch.indices.exists(index=index_name):
                opensearch.indices.create(index=index_name)
            
            # Insérer les documents
            for doc in data["documents"]:
                opensearch.index(index=index_name, id=doc["id"], body=doc["source"])
            
            opensearch.indices.refresh(index=index_name)
            
            await broadcast_log(f"✓ OpenSearch: {len(data['documents'])} documents importés dans '{index_name}'", "success")
            return {"status": "success", "type": "opensearch", "count": len(data["documents"])}
            
        except Exception as e:
            await broadcast_log(f"❌ OpenSearch import error: {e}", "error")
            raise HTTPException(status_code=500, detail=str(e))
    
    else:
        raise HTTPException(status_code=400, detail="Format de fichier non reconnu")


def get_backup_dir():
    """Retourne le chemin du dossier de backup selon l'environnement"""
    import platform
    if platform.system() == "Windows":
        return Path("C:/RAG/backups")
    else:
        return Path("/app/backups")


@app.get("/api/backups")
async def list_backups():
    """Liste les fichiers de backup disponibles"""
    backup_dir = get_backup_dir()
    if not backup_dir.exists():
        return {"backups": [], "backup_dir": str(backup_dir)}
    
    backups = []
    for f in backup_dir.glob("*.json"):
        backups.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime
        })
    
    return {"backups": sorted(backups, key=lambda x: x["modified"], reverse=True)}


@app.get("/api/tracking/{index_name}")
async def get_tracking_stats(index_name: str):
    """Documents indexes pour cet index, d'apres le watcher.

    Remplace IngestionTracker de la V3, qui n'existe pas en V4 : c'est le
    watcher qui tient desormais l'etat de ce qui est indexe.
    """
    etat = await _watcher("/api/watcher/status")
    documents = (etat.get("documents") or {}) if isinstance(etat, dict) else {}
    return {
        "index_name": index_name,
        "files_indexed": documents.get("indexed", 0),
        "deleted": documents.get("deleted", 0),
        "error": etat.get("error") if isinstance(etat, dict) else None,
    }


@app.get("/logo.png")
async def get_logo():
    """Retourne le logo Luciole"""
    pics_logo = PICS_DIR / "luciole-logo.png"
    if pics_logo.exists():
        return FileResponse(str(pics_logo), media_type="image/png")
    if os.path.exists(LOGO_PATH):
        return FileResponse(LOGO_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo non trouvé")


@app.get("/logo.svg")
async def get_logo_svg():
    """Retourne le logo Luciole en SVG"""
    svg_path = PICS_DIR / "luciole.svg"
    if svg_path.exists():
        return FileResponse(str(svg_path), media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Logo SVG non trouvé")


@app.get("/favicon.ico")
@app.get("/favicon.png")
async def get_favicon():
    """Meme favicon que le Chat : pics/favicon.png puis static, puis fallback."""
    for candidate in (
        PICS_DIR / "favicon.png",
        Path(FAVICON_PATH),
        PICS_DIR / "luciole-logo.png",
        Path(LOGO_PATH),
        PICS_DIR / "luciole.png",
    ):
        if candidate.exists():
            return FileResponse(str(candidate), media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon non trouvée")


# ============================================================================
# Configuration Management API
# ============================================================================

def _find_config_file(filename: str) -> str:
    """Find a config file in known locations."""
    candidates = [
        os.path.join("config", filename),
        os.path.join("/app", "config", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _find_ragas_db() -> str:
    """Find the RAGAS SQLite database."""
    candidates = [
        os.path.join("feedbacks", "ragas.db"),
        os.path.join("/app", "feedbacks", "ragas.db"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _modeles_ragas(reglages: dict) -> tuple:
    """Modeles d'evaluation et d'embedding pour RAGAS.

    Ordre : section `ragas` de settings.yaml, sinon le modele de
    generation de l'instance (`llm.model`) — toujours present, puisque
    c'est lui qui repond aux questions. Le nom code en dur de la V3
    (qwen3-30b-a3b-instruct, servi par TensorRT-LLM) n'existe sur aucune
    instance V4 et faisait echouer l'evaluation en quelques secondes.
    """
    ragas_cfg = (reglages or {}).get("ragas") or {}
    defaut = ((reglages or {}).get("llm") or {}).get("model") or "qwen2.5:14b-instruct-q4_K_M"
    return (ragas_cfg.get("eval_model") or defaut,
            ragas_cfg.get("embed_model") or "nomic-embed-text")


def _find_settings_yaml() -> str:
    """Find settings.yaml in known locations."""
    return _find_config_file("settings.yaml")


@app.get("/api/admin/settings")
async def get_settings():
    """Read settings.yaml and return as JSON."""
    filepath = _find_config_file("settings.yaml")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="settings.yaml not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/settings")
async def post_settings(request: Request):
    """Write settings.yaml from JSON body."""
    try:
        data = await request.json()
        filepath = _find_config_file("settings.yaml")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        return JSONResponse(content={"status": "ok", "message": "settings.yaml saved"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/prompts")
async def get_prompts():
    """Read prompts.yaml and return as JSON."""
    filepath = _find_config_file("prompts.yaml")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="prompts.yaml not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/prompts")
async def post_prompts(request: Request):
    """Write prompts.yaml from JSON body."""
    try:
        data = await request.json()
        filepath = _find_config_file("prompts.yaml")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        return JSONResponse(content={"status": "ok", "message": "prompts.yaml saved"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/synonyms")
async def get_synonyms():
    """Read synonyms.txt and return as JSON with text content."""
    filepath = _find_config_file("synonyms.txt")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return JSONResponse(content={"content": content})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="synonyms.txt not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/synonyms")
async def post_synonyms(request: Request):
    """Write synonyms.txt from JSON body (expects {"content": "..."})."""
    try:
        data = await request.json()
        content = data.get("content", "")
        filepath = _find_config_file("synonyms.txt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return JSONResponse(content={"status": "ok", "message": "synonyms.txt saved"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/admin/restart")
async def post_restart():
    """Signal that a manual restart is needed to apply changes."""
    return JSONResponse(content={
        "status": "ok",
        "message": "Configuration updated. Please restart the service manually to apply changes.",
    })


# ============================================================================
# RAGAS Metrics API
# ============================================================================

@app.get("/api/admin/ragas/scores")
async def get_ragas_scores():
    """Read RAGAS scores from feedbacks/ragas.db."""
    db_path = _find_ragas_db()
    if not os.path.exists(db_path):
        return JSONResponse(content={"scores": [], "count": 0})
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM ragas_scores ORDER BY rowid DESC LIMIT 200"
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return JSONResponse(content={"scores": rows, "count": len(rows)})
    except sqlite3.OperationalError:
        return JSONResponse(content={"scores": [], "count": 0})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/ragas/summary")
async def get_ragas_summary():
    """Aggregate RAGAS metrics over the last 30 days."""
    db_path = _find_ragas_db()
    if not os.path.exists(db_path):
        return JSONResponse(content={"summary": {"total_evaluations": 0}})
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT "
            "  AVG(faithfulness) as avg_faithfulness, "
            "  AVG(answer_relevancy) as avg_answer_relevancy, "
            "  AVG(context_recall) as avg_context_recall, "
            "  COUNT(*) as total_evaluations "
            "FROM ragas_scores "
            "WHERE timestamp >= datetime('now', '-30 days')"
        )
        row = cur.fetchone()
        summary = dict(row) if row else {"total_evaluations": 0}
        conn.close()
        return JSONResponse(content={"summary": summary})
    except sqlite3.OperationalError:
        return JSONResponse(content={"summary": {"total_evaluations": 0}})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/ragas/history")
async def get_query_history():
    """Historique des evaluations, lu dans ragas.db.

    La V3 interrogeait d'abord /api/query-history de l'agent ; le coeur
    RAG V4 ne tient pas d'historique, la base RAGAS est donc la seule
    source.
    """
    db_path = _find_ragas_db()
    if not os.path.exists(db_path):
        return JSONResponse(content={"queries": []})
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, timestamp, question, answer, contexts, index_name "
            "FROM query_history ORDER BY id DESC LIMIT 50"
        ).fetchall()
        conn.close()
        return JSONResponse(content={"queries": [dict(r) for r in rows]})
    except sqlite3.OperationalError:
        return JSONResponse(content={"queries": []})


# ============================================================================
# Agents (profils, editeur YAML, traces, escalades) - proxy vers l'agent
# ============================================================================

















class RagasEvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list
    index_name: str = "documents"


@app.post("/api/admin/ragas/evaluate")
async def evaluate_ragas(req: RagasEvalRequest):
    """Trigger RAGAS evaluation on a single question/answer/contexts tuple."""
    import asyncio
    ollama_url = os.environ.get("LLM_URL", os.environ.get("OLLAMA_URL", "http://ollama:11434"))
    try:
        settings_path = _find_settings_yaml()
        with open(settings_path) as f:
            settings = yaml.safe_load(f)
        eval_model, embed_model = _modeles_ragas(settings)
    except Exception:
        eval_model, embed_model = _modeles_ragas({})

    db_path = _find_ragas_db()
    try:
        from evaluation.ragas_evaluator import LucioleRAGASEvaluator
        evaluator = LucioleRAGASEvaluator(
            ollama_url=ollama_url,
            model=eval_model,
            embed_model=embed_model,
            db_path=db_path
        )
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            evaluator.evaluate_single,
            req.question, req.answer, req.contexts, req.index_name
        )
        import math
        def _safe(v):
            if v is None:
                return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return round(v, 4)

        result_scores = {}
        for key in ("faithfulness", "answer_relevancy", "context_recall"):
            val = _safe(scores.get(key))
            if val is not None:
                result_scores[key] = val

        has_any = any(v is not None for v in result_scores.values())
        return JSONResponse(content={
            "status": "ok" if has_any else "partial",
            "scores": result_scores,
            "question": req.question[:200],
            "warning": None if has_any else "Le LLM n'a pas pu produire un format exploitable. Reessayez.",
        })
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"RAGAS dependencies missing: {e}")
    except Exception as e:
        logger.error(f"RAGAS evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/ragas/clear")
async def clear_ragas_history():
    """Purge all RAGAS scores and query history."""
    db_path = _find_ragas_db()
    if not os.path.exists(db_path):
        return JSONResponse(content={"status": "ok", "deleted_scores": 0, "deleted_history": 0})
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM ragas_scores")
        n_scores = cur.fetchone()[0]
        cur.execute("DELETE FROM ragas_scores")
        n_history = 0
        try:
            cur.execute("SELECT count(*) FROM query_history")
            n_history = cur.fetchone()[0]
            cur.execute("DELETE FROM query_history")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
        logger.info(f"RAGAS history cleared: {n_scores} scores + {n_history} queries deleted")
        return JSONResponse(content={"status": "ok", "deleted_scores": n_scores, "deleted_history": n_history})
    except Exception as e:
        logger.error(f"Failed to clear RAGAS history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/ragas/datasets")
async def list_ragas_datasets():
    """List available RAGAS datasets from evaluation/datasets/."""
    datasets_dirs = [
        os.path.join("evaluation", "datasets"),
        os.path.join("/app", "evaluation", "datasets"),
    ]
    results = []
    for d in datasets_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".json"):
                    filepath = os.path.join(d, f)
                    try:
                        with open(filepath, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        results.append({
                            "filename": f,
                            "path": filepath,
                            "name": data.get("name", f),
                            "description": data.get("description", ""),
                            "pairs_count": len(data.get("pairs", [])),
                        })
                    except Exception:
                        pass
            break
    return JSONResponse(content={"datasets": results})


class RagasBatchRequest(BaseModel):
    dataset_path: str
    index_name: str = "documents"


@app.post("/api/admin/ragas/batch")
async def batch_evaluate_ragas(req: RagasBatchRequest):
    """Run RAGAS evaluation on a Q/R dataset: query RAG for each Q, then evaluate."""
    import asyncio
    import httpx

    if not os.path.exists(req.dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.dataset_path}")

    with open(req.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    pairs = dataset.get("pairs", [])
    if not pairs:
        raise HTTPException(status_code=400, detail="Dataset contains no Q/R pairs")

    ollama_url = os.environ.get("LLM_URL", os.environ.get("OLLAMA_URL", "http://ollama:11434"))
    try:
        settings_path = _find_settings_yaml()
        with open(settings_path) as f:
            settings = yaml.safe_load(f)
        eval_model = _modeles_ragas(settings)[0]
    except Exception:
        eval_model = _modeles_ragas({})[0]

    db_path = _find_ragas_db()

    results = []
    for i, pair in enumerate(pairs):
        question = pair.get("question", "")
        ground_truth = pair.get("ground_truth", "")
        if not question:
            continue

        logger.info(f"RAGAS batch [{i+1}/{len(pairs)}]: {question[:80]}...")

        rag_answer = ""
        contexts = []
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{RAG_URL}/api/rag/query",
                    json={"question": question, "deep": False, "history": []}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    rag_answer = data.get("answer", "")
                    # Contextes RAGAS = les passages soumis au modele,
                    # pas les citations : faithfulness et context recall
                    # se mesurent sur ce qui a ete lu.
                    sources = data.get("passages", [])
                    contexts = [p.get("text", "") for p in sources if p.get("text")]
                else:
                    logger.warning(f"Agent query failed (HTTP {resp.status_code})")
        except Exception as e:
            logger.warning(f"Agent query error: {e}")

        if not rag_answer:
            results.append({"question": question[:100], "status": "skip", "reason": "no RAG answer"})
            continue

        try:
            from evaluation.ragas_evaluator import LucioleRAGASEvaluator
            evaluator = LucioleRAGASEvaluator(
                ollama_url=ollama_url, model=eval_model, db_path=db_path
            )
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, evaluator.evaluate_single,
                question, rag_answer, contexts, req.index_name, ground_truth
            )
            results.append({
                "question": question[:100],
                "status": "ok",
                "rag_answer": rag_answer[:200],
                "ground_truth": ground_truth[:200],
                "scores": {
                    "faithfulness": scores.get("faithfulness"),
                    "answer_relevancy": scores.get("answer_relevancy"),
                    "context_recall": scores.get("context_recall"),
                },
            })
        except Exception as e:
            logger.error(f"RAGAS eval error for Q{i+1}: {e}")
            results.append({"question": question[:100], "status": "error", "reason": str(e)})

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    return JSONResponse(content={
        "status": "completed",
        "total": len(pairs),
        "evaluated": ok_count,
        "results": results,
    })


@app.post("/api/admin/ragas/evaluate-feedbacks")
async def evaluate_feedbacks_ragas():
    """Evalue via RAGAS tous les feedbacks 'down' avec correction (expected_response)
    qui n'ont pas encore ete evalues. Utilise expected_response comme ground_truth."""
    import asyncio
    import httpx

    feedback_db = os.environ.get("FEEDBACK_DB_PATH", "/app/feedbacks/feedbacks.db")
    if not os.path.exists(feedback_db):
        raise HTTPException(status_code=404, detail="Feedback database not found")

    conn = sqlite3.connect(feedback_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, query, response, sources, index_name, expected_response "
        "FROM feedbacks WHERE feedback='down' AND expected_response IS NOT NULL "
        "AND expected_response != ''"
    ).fetchall()
    conn.close()

    if not rows:
        return JSONResponse(content={"status": "completed", "total": 0, "evaluated": 0, "results": [], "message": "Aucun feedback negatif avec correction a evaluer"})

    ollama_url = os.environ.get("LLM_URL", os.environ.get("OLLAMA_URL", "http://ollama:11434"))
    try:
        settings_path = _find_settings_yaml()
        with open(settings_path) as f:
            settings = yaml.safe_load(f)
        eval_model, embed_model = _modeles_ragas(settings)
    except Exception:
        eval_model, embed_model = _modeles_ragas({})

    db_path = _find_ragas_db()

    results = []
    for i, row in enumerate(rows):
        question = row["query"]
        ground_truth = row["expected_response"]
        index_name = row["index_name"] or "documents"
        logger.info(f"RAGAS feedback eval [{i+1}/{len(rows)}]: fb#{row['id']} q='{question[:60]}'...")

        rag_answer = ""
        contexts = []
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{RAG_URL}/api/rag/query",
                    json={"question": question, "deep": False, "history": []}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    rag_answer = data.get("answer", "")
                    # Contextes RAGAS = les passages soumis au modele,
                    # pas les citations : faithfulness et context recall
                    # se mesurent sur ce qui a ete lu.
                    sources = data.get("passages", [])
                    contexts = [p.get("text", "") for p in sources if isinstance(p, dict) and p.get("text")]
        except Exception as e:
            logger.warning(f"Agent query error for fb#{row['id']}: {e}")

        if not rag_answer:
            results.append({"feedback_id": row["id"], "question": question[:100], "status": "skip", "reason": "no RAG answer"})
            continue

        try:
            from evaluation.ragas_evaluator import LucioleRAGASEvaluator
            evaluator = LucioleRAGASEvaluator(
                ollama_url=ollama_url, model=eval_model, embed_model=embed_model, db_path=db_path
            )
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, evaluator.evaluate_single,
                question, rag_answer, contexts, index_name, ground_truth
            )
            results.append({
                "feedback_id": row["id"],
                "question": question[:100],
                "ground_truth": ground_truth[:200],
                "rag_answer": rag_answer[:200],
                "status": "ok",
                "scores": {
                    "faithfulness": scores.get("faithfulness"),
                    "answer_relevancy": scores.get("answer_relevancy"),
                    "context_recall": scores.get("context_recall"),
                },
            })
        except Exception as e:
            logger.error(f"RAGAS eval error for fb#{row['id']}: {e}")
            results.append({"feedback_id": row["id"], "question": question[:100], "status": "error", "reason": str(e)})

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    analysis = _build_ragas_analysis(results)
    return JSONResponse(content={
        "status": "completed",
        "total": len(rows),
        "evaluated": ok_count,
        "results": results,
        "analysis": analysis,
    })


def _load_current_config() -> dict:
    """Charge la config courante depuis settings.yaml et prompts.yaml."""
    cfg = {}
    try:
        sp = _find_config_file("settings.yaml")
        with open(sp, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    prompt_text = ""
    try:
        pp = _find_config_file("prompts.yaml")
        with open(pp, "r", encoding="utf-8") as f:
            pd = yaml.safe_load(f) or {}
            prompt_text = pd.get("system_prompt", "")
    except Exception:
        pass
    cfg["_system_prompt"] = prompt_text
    return cfg


def _build_ragas_analysis(results: list) -> dict:
    """Analyse les scores RAGAS, lit la config courante, et propose des modifications concretes."""
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        return {"summary": "Aucun resultat a analyser.", "recommendations": [], "scores_avg": {}}

    cfg = _load_current_config()
    llm = cfg.get("llm", {})
    retrieval = cfg.get("retrieval", {})
    chunking = cfg.get("chunking", {})
    reranker = cfg.get("reranker", {})
    prompt = cfg.get("_system_prompt", "")

    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    thresholds = {"faithfulness": 0.80, "answer_relevancy": 0.75, "context_recall": 0.70}

    avgs = {}
    lows = {}
    for m in metrics:
        vals = [r["scores"].get(m) for r in ok_results
                if r["scores"].get(m) is not None and not (isinstance(r["scores"].get(m), float) and r["scores"][m] != r["scores"][m])]
        if vals:
            avgs[m] = round(sum(vals) / len(vals), 3)
            lows[m] = [r for r in ok_results if r["scores"].get(m) is not None and r["scores"][m] < thresholds[m]]
        else:
            avgs[m] = None
            lows[m] = []

    n = len(ok_results)
    recs = []

    cur_temp = llm.get("temperature", 0.0)
    cur_model = llm.get("model", "?")
    cur_max_tokens = llm.get("max_tokens", 4096)
    cur_rerank_top = retrieval.get("rerank_top_n", 15)
    cur_bm25_top = retrieval.get("bm25_top_k", 40)
    cur_dense_top = retrieval.get("dense_top_k", 40)
    cur_fusion_top = retrieval.get("fusion_top_k", 30)
    cur_bm25_w = retrieval.get("bm25_weight", 0.5)
    cur_dense_w = retrieval.get("dense_weight", 0.5)
    cur_chunk_size = chunking.get("chunk_size", 800)
    cur_chunk_overlap = chunking.get("chunk_overlap", 100)
    cur_reranker_model = reranker.get("model", "?")

    # ── Faithfulness ──
    faith_avg = avgs.get("faithfulness")
    if faith_avg is not None and faith_avg < thresholds["faithfulness"]:
        pct = round(len(lows["faithfulness"]) / n * 100)
        sug_temp = max(0.0, cur_temp - 0.05) if cur_temp > 0.05 else 0.0
        sug_rerank = min(cur_rerank_top + 5, 25)
        has_strict_prompt = any(kw in prompt.lower() for kw in ["uniquement", "exclusivement", "seulement a partir", "only from"])

        param_changes = [
            {"param": "llm.temperature", "file": "settings.yaml", "current": cur_temp, "suggested": sug_temp,
             "reason": "Reduire la creativite du LLM pour limiter les hallucinations"},
            {"param": "retrieval.rerank_top_n", "file": "settings.yaml", "current": cur_rerank_top, "suggested": sug_rerank,
             "reason": "Fournir plus de passages re-classes au LLM pour qu'il trouve l'info dans les sources"},
        ]
        actions = []
        if not has_strict_prompt:
            actions.append("PROMPT : Ajouter une consigne stricte dans le system prompt. Exemple : 'Reponds UNIQUEMENT a partir des passages fournis. Si l'information n'est pas presente, indique-le explicitement.' (via Config UI > System Prompt)")
        if ":8b" in cur_model or ":7b" in cur_model:
            actions.append(f"MODELE : Le modele actuel ({cur_model}) est un petit modele. Les modeles plus gros (14b+) hallucinent significativement moins. Tester qwen2.5:14b ou llama3.1:70b si le GPU le permet (via Config UI > Modele LLM — lecture seule, modèle fixé au démarrage TensorRT-LLM).")
        actions.append(f"TEMPERATURE : Passer de {cur_temp} a {sug_temp} dans settings.yaml (section llm > temperature)")
        actions.append(f"RERANKER : Augmenter rerank_top_n de {cur_rerank_top} a {sug_rerank} pour classer plus de passages")

        recs.append({
            "severity": "high" if faith_avg < 0.6 else "medium",
            "metric": "Fidelite (Faithfulness)",
            "score_avg": faith_avg, "threshold": thresholds["faithfulness"], "failing_pct": pct,
            "diagnostic": f"Le LLM ({cur_model}, temperature={cur_temp}) genere des informations non presentes dans les passages retrouves. {pct}% des reponses evaluees contiennent des hallucinations.",
            "actions": actions,
            "param_changes": param_changes,
        })

    # ── Answer Relevancy ──
    rel_avg = avgs.get("answer_relevancy")
    if rel_avg is not None and rel_avg < thresholds["answer_relevancy"]:
        pct = round(len(lows["answer_relevancy"]) / n * 100)
        sug_max_tokens = max(cur_max_tokens, 4096)

        param_changes = [
            {"param": "llm.max_tokens", "file": "settings.yaml", "current": cur_max_tokens, "suggested": sug_max_tokens,
             "reason": "Augmenter si les reponses sont tronquees avant d'etre completes"},
        ]
        actions = [
            f"PROMPT : Ajouter dans le system prompt une consigne de precision. Exemple : 'Reponds directement a la question posee, de maniere precise et structuree.' (via Config UI > System Prompt)",
            f"MAX_TOKENS : Actuellement {cur_max_tokens}. Si les reponses semblent tronquees, augmenter a {sug_max_tokens}.",
            "SYNONYMES : Ajouter des synonymes metier dans Config UI > synonyms.txt pour que le systeme comprenne les termes specifiques a votre domaine.",
        ]

        recs.append({
            "severity": "high" if rel_avg < 0.5 else "medium",
            "metric": "Pertinence (Answer Relevancy)",
            "score_avg": rel_avg, "threshold": thresholds["answer_relevancy"], "failing_pct": pct,
            "diagnostic": f"Les reponses du LLM ({cur_model}) ne correspondent pas bien aux questions posees. {pct}% des reponses sont jugees hors-sujet ou trop vagues.",
            "actions": actions,
            "param_changes": param_changes,
        })

    # ── Context Recall ──
    recall_avg = avgs.get("context_recall")
    if recall_avg is not None and recall_avg < thresholds["context_recall"]:
        pct = round(len(lows["context_recall"]) / n * 100)
        sug_bm25_top = min(cur_bm25_top + 20, 80)
        sug_dense_top = min(cur_dense_top + 20, 80)
        sug_fusion_top = min(cur_fusion_top + 10, 50)
        sug_chunk_size = max(cur_chunk_size - 200, 300) if cur_chunk_size > 500 else cur_chunk_size
        sug_chunk_overlap = max(int(sug_chunk_size * 0.15), 50)
        sug_bm25_w = min(cur_bm25_w + 0.05, 0.60)
        sug_dense_w = round(1.0 - sug_bm25_w, 2)

        param_changes = [
            {"param": "retrieval.bm25_top_k", "file": "settings.yaml", "current": cur_bm25_top, "suggested": sug_bm25_top,
             "reason": "Elargir le pool de resultats BM25 pour trouver plus de passages candidats"},
            {"param": "retrieval.dense_top_k", "file": "settings.yaml", "current": cur_dense_top, "suggested": sug_dense_top,
             "reason": "Elargir le pool de resultats denses (semantiques)"},
            {"param": "retrieval.fusion_top_k", "file": "settings.yaml", "current": cur_fusion_top, "suggested": sug_fusion_top,
             "reason": "Garder plus de resultats apres la fusion BM25+dense"},
            {"param": "chunking.chunk_size", "file": "settings.yaml", "current": cur_chunk_size, "suggested": sug_chunk_size,
             "reason": "Des chunks plus petits sont plus precis pour la recherche semantique"},
            {"param": "chunking.chunk_overlap", "file": "settings.yaml", "current": cur_chunk_overlap, "suggested": sug_chunk_overlap,
             "reason": "Ajuster le chevauchement proportionnellement a la taille des chunks"},
            {"param": "retrieval.bm25_weight", "file": "settings.yaml", "current": cur_bm25_w, "suggested": sug_bm25_w,
             "reason": "Renforcer la recherche par mots-cles exacts (utile pour termes techniques)"},
            {"param": "retrieval.dense_weight", "file": "settings.yaml", "current": cur_dense_w, "suggested": sug_dense_w,
             "reason": "Ajuster pour compenser le changement de bm25_weight"},
        ]
        actions = [
            f"RECHERCHE : Elargir la recherche : bm25_top_k {cur_bm25_top} -> {sug_bm25_top}, dense_top_k {cur_dense_top} -> {sug_dense_top}, fusion_top_k {cur_fusion_top} -> {sug_fusion_top}",
            f"CHUNKS : Reduire la taille des chunks : {cur_chunk_size} -> {sug_chunk_size} avec overlap {cur_chunk_overlap} -> {sug_chunk_overlap}. Necessite une RE-INGESTION des documents.",
            f"POIDS : Ajuster les poids de fusion : bm25_weight {cur_bm25_w} -> {sug_bm25_w}, dense_weight {cur_dense_w} -> {sug_dense_w}",
            "SYNONYMES : Ajouter des synonymes metier dans Config UI > synonyms.txt pour couvrir les variantes de formulation.",
            "RE-INGESTION : Si les documents ont ete modifies ou si les chunks sont reconfigures, relancer l'ingestion depuis l'onglet Ingestion.",
            "OCR : Si les documents sont des PDFs scannes, verifier la qualite OCR. Des erreurs d'OCR empechent la recherche de trouver les passages.",
        ]

        recs.append({
            "severity": "high" if recall_avg < 0.4 else "medium",
            "metric": "Rappel contextuel (Context Recall)",
            "score_avg": recall_avg, "threshold": thresholds["context_recall"], "failing_pct": pct,
            "diagnostic": f"Le systeme de recherche (BM25 top_k={cur_bm25_top}, dense top_k={cur_dense_top}, chunks={cur_chunk_size}) ne retrouve pas les bons passages. {pct}% des questions n'obtiennent pas les documents pertinents.",
            "actions": actions,
            "param_changes": param_changes,
        })

    # ── Tout va bien ──
    if not recs:
        summary = f"Tous les scores sont au-dessus des seuils. Le systeme fonctionne bien sur les {n} feedbacks evalues."
        recs.append({
            "severity": "info",
            "metric": "Systeme global",
            "diagnostic": f"Les metriques RAGAS sont satisfaisantes avec la config actuelle : modele {cur_model}, temperature {cur_temp}, chunks {cur_chunk_size}. Continuez a collecter des feedbacks pour un suivi dans le temps.",
            "actions": [
                "Continuez a faire tester par les key users pour accumuler plus de donnees.",
                "Comparez ces scores avec une evaluation future apres des modifications (prompt, modele, index).",
            ],
            "param_changes": [],
        })
    else:
        worst = max(recs, key=lambda r: {"high": 2, "medium": 1, "info": 0}[r["severity"]])
        summary = f"Diagnostic sur {n} feedbacks evalues. Probleme principal : {worst['metric']} (moyenne {worst['score_avg']:.2f}, seuil {worst['threshold']:.2f})."

    return {
        "summary": summary,
        "scores_avg": avgs,
        "thresholds": thresholds,
        "total_evaluated": n,
        "current_config": {
            "llm_model": cur_model, "llm_temperature": cur_temp, "llm_max_tokens": cur_max_tokens,
            "bm25_top_k": cur_bm25_top, "dense_top_k": cur_dense_top, "fusion_top_k": cur_fusion_top,
            "bm25_weight": cur_bm25_w, "dense_weight": cur_dense_w,
            "rerank_top_n": cur_rerank_top, "reranker_model": cur_reranker_model,
            "chunk_size": cur_chunk_size, "chunk_overlap": cur_chunk_overlap,
        },
        "recommendations": recs,
    }


@app.post("/api/admin/ragas/simulate")
async def simulate_ragas_analysis():
    """Simule une analyse RAGAS avec des scores fictifs pour demontrer le systeme de recommandations."""
    mock_results = [
        {"feedback_id": 101, "question": "Quels sont les seuils de bruit reglementaires en zone residentielle ?",
         "status": "ok", "scores": {"faithfulness": 0.45, "answer_relevancy": 0.62, "context_recall": 0.35}},
        {"feedback_id": 102, "question": "Comment calculer la surface de plancher selon le PLU ?",
         "status": "ok", "scores": {"faithfulness": 0.72, "answer_relevancy": 0.58, "context_recall": 0.50}},
        {"feedback_id": 103, "question": "Quelle est la procedure de declaration prealable pour une cloture ?",
         "status": "ok", "scores": {"faithfulness": 0.55, "answer_relevancy": 0.70, "context_recall": 0.42}},
        {"feedback_id": 104, "question": "Quel est le delai d'instruction pour un permis de construire ?",
         "status": "ok", "scores": {"faithfulness": 0.90, "answer_relevancy": 0.85, "context_recall": 0.80}},
        {"feedback_id": 105, "question": "Les panneaux photovoltaiques sont-ils soumis a autorisation ?",
         "status": "ok", "scores": {"faithfulness": 0.60, "answer_relevancy": 0.55, "context_recall": 0.30}},
        {"feedback_id": 106, "question": "Quelles sont les obligations de l'exploitant en matiere de gestion des eaux pluviales ?",
         "status": "ok", "scores": {"faithfulness": 0.38, "answer_relevancy": 0.48, "context_recall": 0.25}},
    ]
    analysis = _build_ragas_analysis(mock_results)
    return JSONResponse(content={
        "status": "completed",
        "total": len(mock_results),
        "evaluated": len(mock_results),
        "results": mock_results,
        "analysis": analysis,
        "_note": "Ceci est une SIMULATION avec des scores fictifs pour demontrer le systeme de recommandations.",
    })


# ============================================================================
# Interface HTML
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def admin_ui():
    """Page principale de l'interface d'administration"""
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Luciole RAG - Ingestion</title>
    <link rel="icon" type="image/png" href="/favicon.png">
    <link rel="shortcut icon" type="image/png" href="/favicon.png">
    <style>
        /* Fonts en local -- pas de dependance externe (100% offline) */
        @font-face {
            font-family: 'Segoe UI';
            font-style: normal;
            font-weight: 400 700;
            font-display: swap;
            src: local('Segoe UI'), local('system-ui'), local('-apple-system');
        }
        :root {
            --bg-primary: #0B1929;
            --bg-secondary: #0F2237;
            --bg-tertiary: #163050;
            --bg-terminal: #0B1929;
            --bg-card: #0F2237;
            --bg-elevated: #163050;
            --text-primary: #F8F7F1;
            --text-secondary: #7B96B2;
            --text-bright: #F8F7F1;
            --accent: #FFD76F;
            --accent-dim: #C4952C;
            --accent-glow: rgba(255, 215, 111, 0.25);
            --success: #34D399;
            --warning: #FFD76F;
            --error: #F87171;
            --info: #7B96B2;
            --debug: #7B96B2;
            --border: #1E3A56;
            --log-timestamp: #34D399;
            --log-level-info: #7B96B2;
            --log-level-debug: #7B96B2;
            --log-level-success: #34D399;
            --log-level-warning: #FFD76F;
            --log-level-error: #F87171;
            --log-module: #C4952C;
            --log-function: #FFD76F;
            --log-line: #34D399;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(160deg, #070E18 0%, var(--bg-primary) 40%, #0D1F35 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        /* Layout principal en deux colonnes */
        .app-container {
            display: grid;
            grid-template-columns: 400px 1fr;
            min-height: 100vh;
        }
        
        /* Panneau gauche - Contrôles */
        .left-panel {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border-right: 1px solid var(--border);
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel-header {
            text-align: center;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border);
        }
        
        .panel-header h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #FFD76F, #FFF0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 5px;
        }
        
        .panel-header p {
            color: var(--text-secondary);
            font-size: 0.85rem;
        }
        
        /* Stats compactes */
        .stats-row {
            display: flex;
            gap: 15px;
            justify-content: center;
        }
        
        .stat-box {
            background: var(--bg-secondary);
            padding: 12px 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border);
        }
        
        .stat-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }
        
        /* Sections */
        .section {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid var(--border);
        }
        
        .section-title {
            font-size: 0.9rem;
            font-weight: 600;
            background: linear-gradient(135deg, #FFD76F, #FFF0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Inputs */
        .input-field {
            width: 100%;
            padding: 10px 12px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.9rem;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 10px;
        }
        
        .input-field:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        
        .input-field::placeholder {
            color: var(--text-secondary);
        }
        
        /* Boutons */
        .btn-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 0 14px;
            height: 36px;
            min-height: 36px;
            border: none;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            box-sizing: border-box;
        }
        
        .btn-primary,
        .btn-success,
        .btn-danger,
        .btn-warning {
            background: linear-gradient(135deg, var(--accent), #FFCE60);
            color: #0B1929;
            font-weight: 600;
        }
        
        .btn:hover {
            filter: brightness(1.1);
            transform: translateY(-1px);
            box-shadow: 0 2px 10px var(--accent-glow);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-full {
            width: 100%;
            height: 42px;
            min-height: 42px;
            padding: 0 14px;
            font-size: 0.95rem;
            justify-content: center;
        }
        
        /* Files info */
        .files-info {
            background: var(--bg-primary);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
            display: none;
        }
        
        .files-info.visible {
            display: block;
        }
        
        .files-stats {
            display: flex;
            gap: 15px;
            margin-bottom: 8px;
        }
        
        .file-stat {
            display: flex;
            align-items: baseline;
            gap: 5px;
        }
        
        .file-stat-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--success);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .ext-tags {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        
        .ext-tag {
            padding: 3px 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .ext-tag span {
            color: var(--accent);
        }
        
        /* Progress */
        .progress-section {
            margin-top: 10px;
            display: none;
        }
        
        .progress-section.visible {
            display: block;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 6px;
            color: var(--text-secondary);
        }
        
        .progress-bar {
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            transition: width 0.3s;
        }
        
        /* Index details */
        .index-details {
            margin-top: 10px;
            padding: 10px;
            background: var(--bg-primary);
            border-radius: 6px;
            font-size: 0.85rem;
            display: none;
        }
        
        .index-details.visible {
            display: block;
        }
        
        /* Panneau droit - Terminal */
        .right-panel {
            background: var(--bg-terminal);
            display: flex;
            flex-direction: column;
        }
        
        .terminal-header {
            background: rgba(15, 34, 55, 0.8);
            backdrop-filter: blur(12px);
            padding: 10px 15px;
            border-bottom: 1px solid rgba(255, 215, 111, 0.15);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .terminal-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .terminal-dots {
            display: flex;
            gap: 6px;
        }
        
        .terminal-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .dot-red { background: #ff5f56; }
        .dot-yellow { background: #ffbd2e; }
        .dot-green { background: #27c93f; }
        
        .terminal-actions {
            display: flex;
            gap: 10px;
        }
        
        .terminal-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.8rem;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .terminal-btn:hover {
            background: var(--bg-card);
            color: var(--text-primary);
        }
        
        /* Terminal logs - style loguru exact */
        /* Ordre inversé : logs récents en haut */
        .terminal-logs {
            flex: 1;
            overflow-y: auto;
            padding: 5px 0;
            font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            scroll-behavior: smooth;
            display: flex;
            flex-direction: column-reverse;
        }
        
        .log-line {
            padding: 1px 12px;
            white-space: nowrap;
            overflow-x: hidden;
            text-overflow: ellipsis;
        }
        
        .log-line:hover {
            background: rgba(255, 255, 255, 0.05);
            white-space: pre-wrap;
            word-break: break-all;
        }
        
        .log-timestamp {
            color: var(--log-timestamp);
        }
        
        .log-separator {
            color: var(--text-secondary);
        }
        
        .log-level {
            font-weight: 500;
        }
        
        .log-level.INFO { color: var(--log-level-info); }
        .log-level.DEBUG { color: var(--log-level-debug); }
        .log-level.SUCCESS { color: var(--log-level-success); }
        .log-level.WARNING { color: var(--log-level-warning); }
        .log-level.ERROR { color: var(--log-level-error); }
        
        .log-location {
            color: var(--log-module);
        }
        
        .log-message {
            color: var(--text-primary);
        }
        
        .log-message.success { color: var(--success); }
        .log-message.error { color: var(--error); }
        .log-message.warning { color: var(--accent); }
        .log-message.info { color: var(--text-primary); }
        
        /* Barre de progression style tqdm */
        .batch-bar {
            color: var(--success);
            font-weight: 500;
        }
        
        .progress-bar-line {
            background: rgba(78, 201, 176, 0.1);
        }
        
        /* Section paramètres */
        .profile-selector {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        
        .profile-btn {
            flex: 1;
            padding: 0 10px;
            height: 36px;
            min-height: 36px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        }
        
        .profile-btn:hover {
            background: rgba(255, 215, 111, 0.1);
            border-color: var(--accent-dim);
            color: var(--text-primary);
        }
        
        .profile-btn.active {
            background: rgba(255, 215, 111, 0.2);
            border-color: var(--accent);
            color: var(--accent);
            font-weight: 600;
        }
        
        .profile-description {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 10px;
            padding: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            border-left: 3px solid var(--accent);
        }
        
        .params-details {
            font-size: 0.85rem;
        }
        
        .params-details summary {
            cursor: pointer;
            color: var(--text-secondary);
            padding: 5px 0;
        }
        
        .params-details summary:hover {
            color: var(--accent);
        }
        
        .params-details[open] summary {
            margin-bottom: 10px;
        }
        
        .params-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .param-group {
            background: var(--bg-primary);
            padding: 10px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        
        .param-group-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 8px;
        }
        
        .param-group label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }
        
        .param-group input {
            width: 60px;
            padding: 4px 6px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            text-align: right;
        }
        
        .param-group input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        
        /* Hidden */
        .hidden {
            display: none !important;
        }
        
        /* Scrollbar - global */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-dim);
        }
        
        /* Responsive */
        @media (max-width: 900px) {
            .app-container {
                grid-template-columns: 1fr;
                grid-template-rows: auto 1fr;
            }
            
            .left-panel {
                border-right: none;
                border-bottom: 1px solid var(--border);
                max-height: 50vh;
            }
            
            .right-panel {
                min-height: 50vh;
            }
        }
        
        /* ========== Navigation par onglets ========== */
        .top-nav {
            display: flex;
            background: rgba(15, 34, 55, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 215, 111, 0.15);
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
            padding: 0 20px;
            gap: 0;
        }
        
        .nav-tab {
            padding: 12px 24px;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: -2px;
        }
        
        .nav-tab:hover {
            color: var(--text-primary);
            background: rgba(255, 215, 111, 0.05);
        }
        
        .nav-tab.active {
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            font-weight: 600;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        #tab-ingestion.active .app-container {
            min-height: calc(100vh - 46px);
        }
        
        /* ========== Config styles (shared with RAGAS) ========== */
        .config-section-title {
            font-size: 1rem;
            font-weight: 600;
            background: linear-gradient(135deg, #FFD76F, #FFF0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .config-status {
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-top: 10px;
            display: none;
        }
        
        .config-status.success {
            display: block;
            background: rgba(78, 201, 176, 0.1);
            border: 1px solid rgba(78, 201, 176, 0.3);
            color: var(--success);
        }
        
        .config-status.error {
            display: block;
            background: rgba(241, 76, 76, 0.1);
            border: 1px solid rgba(241, 76, 76, 0.3);
            color: var(--error);
        }
        
        /* ========== Analysis panel ========== */
        .analysis-panel {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .analysis-header {
            padding: 15px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .analysis-header.ok { background: rgba(78, 201, 176, 0.1); border-bottom: 1px solid rgba(78, 201, 176, 0.3); }
        .analysis-header.warn { background: rgba(255, 180, 50, 0.1); border-bottom: 1px solid rgba(255, 180, 50, 0.3); }
        .analysis-header.bad { background: rgba(241, 76, 76, 0.1); border-bottom: 1px solid rgba(241, 76, 76, 0.3); }
        .analysis-header h3 { margin: 0; font-size: 1rem; color: var(--text-bright); }
        .analysis-averages {
            display: flex; gap: 20px; padding: 15px 20px;
            background: var(--bg-elevated); border-bottom: 1px solid var(--border);
        }
        .analysis-avg-item {
            display: flex; flex-direction: column; align-items: center; gap: 4px;
        }
        .analysis-avg-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; }
        .analysis-avg-value { font-size: 1.4rem; font-weight: 700; }
        .analysis-avg-value.good { color: var(--success); }
        .analysis-avg-value.warn { color: var(--accent); }
        .analysis-avg-value.bad { color: var(--error); }
        .analysis-avg-threshold { font-size: 0.7rem; color: var(--text-secondary); }
        .analysis-recs { padding: 0; }
        .analysis-rec {
            padding: 15px 20px; border-bottom: 1px solid var(--border);
        }
        .analysis-rec:last-child { border-bottom: none; }
        .analysis-rec-header {
            display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
        }
        .analysis-severity {
            padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        }
        .analysis-severity.high { background: rgba(241, 76, 76, 0.2); color: var(--error); }
        .analysis-severity.medium { background: rgba(255, 215, 111, 0.2); color: var(--accent); }
        .analysis-severity.info { background: rgba(78, 201, 176, 0.2); color: var(--success); }
        .analysis-rec-metric { font-weight: 600; color: var(--text-bright); }
        .analysis-rec-score { font-size: 0.85rem; color: var(--text-secondary); margin-left: auto; }
        .analysis-diagnostic {
            color: var(--text-primary); font-size: 0.9rem; margin-bottom: 10px;
            padding: 10px 12px; background: var(--bg-primary); border-radius: 6px;
        }
        .analysis-actions { list-style: none; padding: 0; margin: 0; }
        .analysis-actions li {
            padding: 6px 0 6px 20px; position: relative;
            color: var(--text-secondary); font-size: 0.85rem; line-height: 1.5;
        }
        .analysis-actions li::before {
            content: ''; position: absolute; left: 4px; top: 13px;
            width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
        }

        /* ========== RAGAS tab ========== */
        .ragas-container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 30px 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .ragas-summary-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        
        .ragas-card {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        
        .ragas-card-value {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 5px;
        }
        
        .ragas-card-value.good { color: var(--success); }
        .ragas-card-value.bad { color: var(--error); }
        
        .ragas-card-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .ragas-card-threshold {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        
        .ragas-table-section {
            background: rgba(15, 34, 55, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            overflow-x: auto;
        }
        
        .ragas-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        
        .ragas-table th {
            text-align: left;
            padding: 10px 12px;
            color: var(--text-secondary);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
            font-size: 0.8rem;
        }
        
        .ragas-table td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(30, 58, 86, 0.5);
            color: var(--text-primary);
        }
        
        .ragas-table tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }
        
        .ragas-score-good {
            color: var(--success);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .ragas-score-bad {
            color: var(--error);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .ragas-refresh-info {
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <!-- Barre de navigation -->
    <div class="top-nav">
        <button class="nav-tab active" onclick="switchTab('ingestion')" id="nav-ingestion">Ingestion</button>
        <button class="nav-tab" onclick="switchTab('ragas')" id="nav-ragas">RAGAS</button>
    </div>

    <!-- Onglet Ingestion (UI existante) -->
    <div id="tab-ingestion" class="tab-content active">
    <div class="app-container">
        <!-- Panneau gauche - Contrôles -->
        <div class="left-panel">
            <div class="panel-header">
                <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                    <img src="/logo.png" alt="Luciole" style="height: 45px; width: auto; filter: drop-shadow(0 0 6px rgba(255,215,111,0.25));">
                    <h1 style="margin: 0;">Luciole RAG</h1>
                </div>
            </div>
            
            <!-- Gestion des Index -->
            <div class="section">
                <div class="section-title">🗄️ Gestion des Index</div>
                
                <select id="index-select" class="input-field">
                    <option value="">-- Sélectionner un index --</option>
                </select>
                
                <div class="btn-row">
                    <button class="btn btn-primary" onclick="refreshIndexes()">Actualiser</button>
                    <button class="btn btn-danger" onclick="deleteSelectedIndex()">Supprimer</button>
                    <button class="btn btn-warning" onclick="clearAllIndexes()">Tout effacer</button>
                </div>
                
                <div class="btn-row" style="margin-top: 10px;">
                    <button class="btn btn-success" onclick="exportSelectedIndex()">📦 Exporter</button>
                    <button class="btn btn-info" onclick="showImportDialog()">📥 Importer</button>
                </div>
                
                <!-- Liste des backups -->
                <div id="backups-section" style="margin-top: 15px; display: none;">
                    <div style="font-size: 12px; color: #888; margin-bottom: 5px;">Backups disponibles:</div>
                    <select id="backup-select" class="input-field" style="font-size: 11px;">
                        <option value="">-- Sélectionner un backup --</option>
                    </select>
                    <div class="btn-row" style="margin-top: 5px;">
                        <button class="btn btn-info btn-sm" onclick="importSelectedBackup()">Importer ce backup</button>
                        <button class="btn btn-secondary btn-sm" onclick="hideImportDialog()">Annuler</button>
                    </div>
                </div>
            </div>
            
            <!-- Paramètres d'ingestion (lecture seule depuis settings.yaml) -->
            <div class="section">
                <div class="section-title">⚙️ Paramètres (depuis settings.yaml)</div>
                
                <div class="profile-description" id="profile-desc" style="margin-bottom: 10px;">
                    📄 Configuration chargée depuis <code>config/settings.yaml</code><br>
                    <small style="color: #888;">Modifiez le fichier YAML et redémarrez les containers pour changer les paramètres.</small>
                </div>
                
                <div class="params-grid">
                    <div class="param-group">
                        <div class="param-group-title">📦 Chunking</div>
                        <label>Chunk Size <input type="number" id="param-chunk-size" value="512" readonly style="background: #222; cursor: not-allowed;"></label>
                        <label>Overlap <input type="number" id="param-overlap" value="50" readonly style="background: #222; cursor: not-allowed;"></label>
                    </div>
                    <div class="param-group">
                        <div class="param-group-title">🔍 Retrieval</div>
                        <label>BM25 Top K <input type="number" id="param-bm25-topk" value="50" readonly style="background: #222; cursor: not-allowed;"></label>
                        <label>Dense Top K <input type="number" id="param-dense-topk" value="50" readonly style="background: #222; cursor: not-allowed;"></label>
                        <label>Fusion Top K <input type="number" id="param-fusion-topk" value="30" readonly style="background: #222; cursor: not-allowed;"></label>
                    </div>
                    <div class="param-group">
                        <div class="param-group-title">🎯 Reranking</div>
                        <label>Rerank Top N <input type="number" id="param-rerank-topn" value="10" readonly style="background: #222; cursor: not-allowed;"></label>
                    </div>
                    <div class="param-group">
                        <div class="param-group-title">🧠 Embedding</div>
                        <label>Batch Size <input type="number" id="param-batch-size" value="32" readonly style="background: #222; cursor: not-allowed;"></label>
                    </div>
                </div>
            </div>
            
            <!-- Watcher : supervision (l'ingestion est automatique) -->
            <div class="section">
                <div class="section-title">👁️ Watcher</div>

                <div style="font-size:12px;color:#999;line-height:1.5;margin-bottom:10px;">
                    L'ingestion est automatique : déposez, modifiez ou retirez
                    un fichier dans <code>data/</code>, le watcher s'en charge.
                    Cette page supervise, elle ne lance rien.
                </div>

                <div class="files-info" style="display:block;">
                    <div class="files-stats">
                        <div class="file-stat">
                            <span class="file-stat-value" id="w-indexed">--</span>
                            <span>indexés</span>
                        </div>
                        <div class="file-stat">
                            <span class="file-stat-value" id="w-jobs">--</span>
                            <span>en attente</span>
                        </div>
                        <div class="file-stat">
                            <span class="file-stat-value" id="w-errors">--</span>
                            <span>erreurs</span>
                        </div>
                    </div>
                    <div id="w-state" style="margin-top:8px;font-size:12px;color:#999;">
                        État inconnu
                    </div>
                </div>

                <div class="btn-row" style="margin-top:10px;">
                    <button class="btn btn-primary" onclick="refreshWatcher()" style="flex:1;">
                        Actualiser
                    </button>
                    <button class="btn btn-warning" onclick="rescanWatcher()" style="flex:1;"
                            title="Compare le disque et l'index, puis enfile les différences">
                        Réconcilier
                    </button>
                </div>
            </div>

        <!-- Panneau droit - Terminal -->
        <div class="right-panel">
            <div class="terminal-header">
                <div class="terminal-title">
                    <div class="terminal-dots">
                        <div class="terminal-dot dot-red"></div>
                        <div class="terminal-dot dot-yellow"></div>
                        <div class="terminal-dot dot-green"></div>
                    </div>
                    <span>Journal du watcher</span>
                </div>
                <div class="terminal-actions">
                    <button class="terminal-btn" onclick="clearLogs()">Effacer</button>
                    <button class="terminal-btn" onclick="scrollToBottom()">Récents</button>
                </div>
            </div>
            
            <div class="terminal-logs" id="logs-container"></div>
        </div>
    </div>
    </div><!-- /tab-ingestion -->

    <!-- Onglet RAGAS -->
    <div id="tab-ragas" class="tab-content">
        <div class="ragas-container">
            <!-- Cartes de resume -->
            <div class="ragas-summary-cards">
                <div class="ragas-card">
                    <div class="ragas-card-value" id="ragas-faithfulness">--</div>
                    <div class="ragas-card-label">Faithfulness</div>
                    <div class="ragas-card-threshold">Seuil: 0.80</div>
                </div>
                <div class="ragas-card">
                    <div class="ragas-card-value" id="ragas-relevancy">--</div>
                    <div class="ragas-card-label">Answer Relevancy</div>
                    <div class="ragas-card-threshold">Seuil: 0.75</div>
                </div>
                <div class="ragas-card">
                    <div class="ragas-card-value" id="ragas-recall">--</div>
                    <div class="ragas-card-label">Context Recall</div>
                    <div class="ragas-card-threshold">Seuil: 0.70</div>
                </div>
            </div>

            <!-- Tableau des evaluations existantes -->
            <div class="ragas-table-section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div class="config-section-title" style="margin-bottom: 0;">Evaluations recentes</div>
                    <div style="display:flex; gap:8px;">
                        <button class="btn btn-primary" onclick="loadRagasData()">Actualiser</button>
                        <button class="btn btn-warning" onclick="clearRagasHistory()">Effacer l'historique</button>
                    </div>
                </div>
                <table class="ragas-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Question</th>
                            <th>Faithfulness</th>
                            <th>Relevancy</th>
                            <th>Context Recall</th>
                        </tr>
                    </thead>
                    <tbody id="ragas-table-body">
                        <tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">Chargement...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Historique des requetes (pour evaluation manuelle) -->
            <div class="ragas-table-section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div class="config-section-title" style="margin-bottom: 0;">Historique des requetes</div>
                    <button class="btn" onclick="loadQueryHistory()">Charger l'historique</button>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 10px;">
                    Cliquez sur "Evaluer" pour lancer une evaluation RAGAS sur une requete passee.
                    L'evaluation prend 30-60 secondes (le LLM local analyse la fidelite, la pertinence et le rappel).
                </p>
                <table class="ragas-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Question</th>
                            <th style="width:80px">Reponse</th>
                            <th style="width:80px">Ctx</th>
                            <th style="width:100px">Action</th>
                        </tr>
                    </thead>
                    <tbody id="ragas-history-body">
                        <tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">Cliquez sur "Charger l'historique"</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Section Evaluation Batch -->
            <div class="ragas-table-section" style="margin-top: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div class="config-section-title" style="margin-bottom: 0;">Evaluation par lot (Dataset Q/R)</div>
                    <button class="btn" onclick="loadRagasDatasets()">Charger les datasets</button>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 10px;">
                    Selectionnez un dataset de questions/reponses de reference. Pour chaque question,
                    Luciole interroge le RAG puis RAGAS evalue la fidelite, la pertinence et le rappel contextuel.
                </p>
                <div id="ragas-datasets-list" style="margin-bottom: 15px;">
                    <em style="color: var(--text-secondary);">Cliquez sur "Charger les datasets" pour voir les fichiers disponibles.</em>
                </div>
                <div id="ragas-batch-status" style="display:none; padding: 15px; background: var(--bg-elevated); border-radius: 8px; margin-top: 10px;">
                    <div id="ragas-batch-progress" style="color: var(--text-primary); font-size: 0.9rem;"></div>
                    <div id="ragas-batch-results" style="margin-top: 10px;"></div>
                </div>
            </div>

            <!-- Section Evaluation des feedbacks -->
            <div class="ragas-table-section" style="margin-top: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div class="config-section-title" style="margin-bottom: 0;">Evaluation des feedbacks (corrections key users)</div>
                    <div style="display:flex;gap:10px">
                        <button class="btn btn-primary" onclick="evaluateFeedbacks()" id="btn-eval-feedbacks">Lancer l'evaluation RAGAS</button>
                        <button class="btn" onclick="simulateRagas()" id="btn-simulate" title="Teste l'analyse avec des scores fictifs">Simulation demo</button>
                    </div>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 10px;">
                    Evalue via RAGAS les feedbacks negatifs avec correction : re-interroge le RAG pour chaque question,
                    puis compare avec la reponse attendue fournie par le key user (ground truth).
                    A lancer periodiquement (hebdomadaire ou mensuel) pour mesurer l'amelioration du systeme.
                </p>
                <div id="feedback-eval-status" style="display:none; padding: 15px; background: var(--bg-elevated); border-radius: 8px; margin-top: 10px;">
                    <div id="feedback-eval-progress" style="color: var(--text-primary); font-size: 0.9rem;"></div>
                    <div id="feedback-eval-results" style="margin-top: 10px;"></div>
                </div>
            </div>
        </div>
    </div><!-- /tab-ragas -->


    
    <script>
        let ws = null;
        let selectedPath = '';
        
        // Initialisation
        document.addEventListener('DOMContentLoaded', () => {
            // Message de bienvenue style terminal
            addLog({
                timestamp: new Date().toISOString(),
                level: 'info',
                message: 'Luciole RAG v1.0 - Interface d\\'ingestion',
                data: {}
            });
            addLog({
                timestamp: new Date().toISOString(),
                level: 'info',
                message: 'Connexion au serveur...',
                data: {}
            });
            
            connectWebSocket();
            loadStats();
            refreshIndexes();
            loadConfigFromYaml();  // Charger les paramètres depuis settings.yaml

            // Watcher : etat et journal, rafraichis par sondage.
            refreshWatcher();
            refreshWatcherLogs();
            setInterval(refreshWatcher, 5000);
            setInterval(refreshWatcherLogs, 5000);

            // Force auto-scroll périodique
            // Avec flex-direction: column-reverse, scrollTop=0 montre les logs récents (en haut)
            setInterval(() => {
                if (autoScroll) {
                    const container = document.getElementById('logs-container');
                    container.scrollTop = 0;
                }
            }, 100);
        });
        
        // WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/logs`);
            
            ws.onmessage = (event) => {
                const log = JSON.parse(event.data);
                addLog(log);
                
                if (log.data && log.data.progress !== undefined) {
                    updateProgress(log.data.progress, log.data.total);
                }
                
                if (log.data && log.data.qdrant_vectors !== undefined) {
                    document.getElementById('stat-qdrant').textContent = log.data.qdrant_vectors;
                    document.getElementById('stat-opensearch').textContent = log.data.opensearch_docs;
                    resetUI();
                }
            };
            
            ws.onclose = () => {
                setTimeout(connectWebSocket, 2000);
            };
        }
        
        // Ouvrir le dialogue de sélection de dossier
        function openFolderDialog() {
            document.getElementById('folder-input').click();
        }
        
        // Quand un dossier est sélectionné via le dialogue
        // ── Watcher ────────────────────────────────────────────────────
        // L'admin ne commande plus l'ingestion (le watcher est la seule
        // voie d'entree en V4) : elle affiche son etat et son journal.

        async function refreshWatcher() {
            try {
                const st = await (await fetch('/api/watcher/status')).json();
                if (st.error) {
                    document.getElementById('w-state').textContent =
                        'Watcher injoignable : ' + (st.detail || st.error);
                    return;
                }
                const docs = st.documents || {};
                const jobs = st.jobs || st.queue || {};
                document.getElementById('w-indexed').textContent = docs.indexed ?? '--';
                document.getElementById('w-jobs').textContent =
                    (jobs.pending ?? 0) + (jobs.in_progress ?? 0);
                document.getElementById('w-errors').textContent = jobs.failed ?? 0;
                const actif = st.running ?? st.watcher?.running;
                document.getElementById('w-state').textContent =
                    actif ? 'Service actif, surveillance en cours' : 'Service arrete';
            } catch (e) {
                document.getElementById('w-state').textContent = 'Etat indisponible : ' + e;
            }
        }

        async function rescanWatcher() {
            const r = await (await fetch('/api/watcher/rescan', {method: 'POST'})).json();
            if (r.error) {
                addLog({timestamp: new Date().toISOString(), level: 'error',
                        message: 'Reconciliation impossible : ' + (r.detail || r.error)});
            } else {
                addLog({timestamp: new Date().toISOString(), level: 'success',
                        message: 'Reconciliation demandee au watcher'});
            }
            setTimeout(refreshWatcher, 2000);
        }

        // Journal du conteneur watcher, rafraichi par sondage : le
        // watcher n'expose pas de flux, et un sondage espace suffit pour
        // suivre une ingestion qui dure des minutes.
        let dernierJournal = '';

        async function refreshWatcherLogs() {
            try {
                const r = await (await fetch('/api/watcher/logs?lines=200')).json();
                if (r.error) {
                    if (dernierJournal !== r.error) {
                        dernierJournal = r.error;
                        addLog({timestamp: new Date().toISOString(), level: 'warning',
                                message: 'Journal indisponible : ' + (r.detail || r.error)});
                    }
                    return;
                }
                const texte = (r.lines || []).join('
');
                if (texte === dernierJournal) return;
                dernierJournal = texte;
                const container = document.getElementById('logs-container');
                container.innerHTML = '';
                (r.lines || []).forEach(ligne => {
                    const div = document.createElement('div');
                    div.className = 'log-line';
                    div.textContent = ligne;
                    container.appendChild(div);
                });
                if (autoScroll) scrollToBottom();
            } catch (e) {
                // Un reseau qui hoquette ne doit pas remplir le terminal.
            }
        }

        // Logs style terminal loguru exact
        let autoScroll = true;
        
        function addLog(log) {
            const container = document.getElementById('logs-container');
            const now = new Date(log.timestamp);
            const timestamp = now.toISOString().replace('T', ' ').substring(0, 23);
            
            const levelMap = {
                'info': 'INFO',
                'success': 'SUCCESS', 
                'warning': 'WARNING',
                'error': 'ERROR',
                'debug': 'DEBUG'
            };
            
            const level = levelMap[log.level] || 'INFO';
            
            // Déterminer le module basé sur le contexte
            let modulePath = 'src.api.admin_ui';
            let funcName = 'broadcast';
            let lineNo = '0';
            
            const msg = log.message;
            
            if (msg.includes('Ingesting file:') || msg.includes('fichier')) {
                modulePath = 'src.ingestion.pipeline';
                funcName = 'ingest_file';
                lineNo = '169';
            } else if (msg.includes('Parsing PDF')) {
                modulePath = 'src.ingestion.parsers';
                funcName = 'parse';
                lineNo = '79';
            } else if (msg.includes('Parsing DOCX')) {
                modulePath = 'src.ingestion.parsers';
                funcName = 'parse';
                lineNo = '165';
            } else if (msg.includes('Parsing XLSX') || msg.includes('Excel')) {
                modulePath = 'src.ingestion.excel_parser';
                funcName = 'parse';
                lineNo = '61';
            } else if (msg.includes('Parsing TXT')) {
                modulePath = 'src.ingestion.parsers';
                funcName = 'parse';
                lineNo = '450';
            } else if (msg.includes('Created') && msg.includes('chunks')) {
                modulePath = 'src.ingestion.chunker';
                funcName = 'chunk';
                lineNo = '91';
            } else if (msg.includes('Generating embeddings') || msg.includes('embeddings')) {
                modulePath = 'src.ingestion.embedder';
                funcName = 'embed_chunks';
                lineNo = '92';
            } else if (msg.includes('Generated') && msg.includes('embeddings')) {
                modulePath = 'src.ingestion.embedder';
                funcName = 'embed_chunks';
                lineNo = '116';
            } else if (msg.includes('Indexed') && msg.includes('Qdrant')) {
                modulePath = 'src.ingestion.pipeline';
                funcName = '_index_qdrant';
                lineNo = '261';
            } else if (msg.includes('Indexed') && msg.includes('OpenSearch')) {
                modulePath = 'src.ingestion.pipeline';
                funcName = '_index_opensearch';
                lineNo = '289';
            } else if (msg.includes('Ingestion complete')) {
                modulePath = 'src.ingestion.pipeline';
                funcName = 'ingest_file';
                lineNo = '197';
            } else if (msg.includes('OCR')) {
                modulePath = 'src.ingestion.parsers';
                funcName = '_apply_ocr';
                lineNo = '135';
            }
            
            const entry = document.createElement('div');
            entry.className = 'log-line';
            
            // Format exact loguru
            const levelPadded = level.padEnd(8);
            const locationStr = `${modulePath}:${funcName}:${lineNo}`;
            
            entry.innerHTML = `<span class="log-timestamp">${timestamp}</span> <span class="log-separator">|</span> <span class="log-level ${level}">${levelPadded}</span><span class="log-separator">|</span> <span class="log-location">${locationStr}</span> <span class="log-separator">-</span> <span class="log-message ${log.level}">${escapeHtml(log.message)}</span>`;
            
            container.appendChild(entry);
            
            // Limiter le nombre de lignes (supprimer les plus anciens = premiers du DOM avec column-reverse)
            while (container.children.length > 2000) {
                container.removeChild(container.firstChild);
            }
            
            // Force auto-scroll (column-reverse: scrollTop=0 = logs récents en haut)
            if (autoScroll) {
                container.scrollTop = 0;
            }
        }
        
        // Ajouter une barre de progression style tqdm
        function addProgressBar(current, total, speed) {
            const container = document.getElementById('logs-container');
            const percent = Math.round((current / total) * 100);
            const barWidth = 40;
            const filled = Math.round((percent / 100) * barWidth);
            const empty = barWidth - filled;
            const bar = '█'.repeat(filled) + ' '.repeat(empty);
            
            // Supprimer l'ancienne barre si elle existe
            const existing = container.querySelector('.progress-bar-line');
            if (existing) {
                existing.remove();
            }
            
            const entry = document.createElement('div');
            entry.className = 'log-line progress-bar-line';
            entry.innerHTML = `<span class="batch-bar">Batches: ${percent}%|${bar}| ${current}/${total} [${speed}]</span>`;
            
            container.appendChild(entry);
            // Avec column-reverse, scrollTop=0 pour voir les nouveaux logs
            if (autoScroll) {
                container.scrollTop = 0;
            }
        }
        
        // Détecter si l'utilisateur scroll manuellement
        // Avec column-reverse: scrollTop=0 = en haut (logs récents)
        document.getElementById('logs-container').addEventListener('scroll', function() {
            const container = this;
            const isAtTop = container.scrollTop <= 50;
            autoScroll = isAtTop;
        });
        
        function clearLogs() {
            const container = document.getElementById('logs-container');
            container.innerHTML = '';
            addLog({
                timestamp: new Date().toISOString(),
                level: 'info',
                message: 'Logs effacés. Prêt.',
                data: {}
            });
        }
        
        function scrollToBottom() {
            // Avec column-reverse, scrollTop=0 = voir les logs récents (en haut)
            const container = document.getElementById('logs-container');
            container.scrollTop = 0;
            autoScroll = true;
        }
        
        // Force scroll au démarrage
        window.addEventListener('load', () => {
            setTimeout(scrollToBottom, 500);
        });
        
        // Stats
        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('stat-qdrant').textContent = data.qdrant_vectors || 0;
                document.getElementById('stat-opensearch').textContent = data.opensearch_documents || 0;
            } catch (e) {
                console.log('Stats not available');
            }
        }
        
        // Gestion des index
        async function refreshIndexes() {
            try {
                const res = await fetch('/api/indexes');
                const data = await res.json();
                
                const select = document.getElementById('index-select');
                select.innerHTML = '<option value="">-- Sélectionner un index --</option>';
                
                // Utiliser la liste unifiée (sans doublons case-insensitive)
                // Fallback sur l'ancienne méthode si unified_indexes n'existe pas
                let indexList = [];
                
                if (data.unified_indexes && data.unified_indexes.length > 0) {
                    // Nouvelle API avec liste unifiée
                    indexList = data.unified_indexes.map(idx => ({
                        name: idx.name,
                        vectors: idx.qdrant_vectors,
                        docs: idx.opensearch_docs
                    }));
                } else {
                    // Fallback: déduplication manuelle case-insensitive
                    const seen = new Map(); // lowercase -> original name
                    
                    if (data.qdrant_collections) {
                        data.qdrant_collections.forEach(c => {
                            const key = c.name.toLowerCase();
                            if (!seen.has(key)) {
                                seen.set(key, c.name);
                            }
                        });
                    }
                    if (data.opensearch_indexes) {
                        data.opensearch_indexes.forEach(i => {
                            const key = i.name.toLowerCase();
                            if (!seen.has(key)) {
                                seen.set(key, i.name);
                            }
                        });
                    }
                    
                    indexList = Array.from(seen.values()).map(name => ({ name }));
                }
                
                indexList.forEach(idx => {
                    const opt = document.createElement('option');
                    opt.value = idx.name;
                    // Afficher le nombre de vecteurs/docs si disponible
                    const info = idx.vectors ? ` (${idx.vectors} vecteurs)` : '';
                    opt.textContent = idx.name + info;
                    select.appendChild(opt);
                });
                
                // Stocker les données pour affichage
                window.indexData = data;
                
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'info',
                    message: `🗄️ ${indexList.length} index trouvés`,
                    data: {}
                });
                
                document.getElementById('index-details').classList.add('visible');
                updateIndexDetails();
                
            } catch (e) {
                console.error(e);
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'error',
                    message: '❌ Erreur lors du chargement des index',
                    data: {}
                });
            }
        }
        
        function updateIndexDetails() {
            const data = window.indexData || {};
            
            let qdrantInfo = '-';
            let osInfo = '-';
            
            if (data.qdrant_collections && data.qdrant_collections.length > 0) {
                const totalVectors = data.qdrant_collections.reduce((sum, c) => sum + (c.vectors_count || 0), 0);
                qdrantInfo = `${data.qdrant_collections.length} collections, ${totalVectors} vecteurs`;
            } else if (data.qdrant_error) {
                qdrantInfo = `Erreur: ${data.qdrant_error}`;
            }
            
            if (data.opensearch_indexes && data.opensearch_indexes.length > 0) {
                const totalDocs = data.opensearch_indexes.reduce((sum, i) => sum + (i.documents_count || 0), 0);
                osInfo = `${data.opensearch_indexes.length} index, ${totalDocs} documents`;
            } else if (data.opensearch_error) {
                osInfo = `Erreur: ${data.opensearch_error}`;
            }
            
            document.getElementById('detail-qdrant').textContent = qdrantInfo;
            document.getElementById('detail-opensearch').textContent = osInfo;
        }
        
        async function deleteSelectedIndex() {
            const select = document.getElementById('index-select');
            const indexName = select.value;
            
            if (!indexName) {
                alert('Veuillez sélectionner un index à supprimer');
                return;
            }
            
            if (!confirm(`Êtes-vous sûr de vouloir supprimer l'index "${indexName}" ?\\nCette action est irréversible.`)) {
                return;
            }
            
            try {
                const res = await fetch(`/api/index/${indexName}`, { method: 'DELETE' });
                const data = await res.json();
                
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'success',
                    message: `✓ Index "${indexName}" supprimé`,
                    data: data
                });
                
                await refreshIndexes();
                await loadStats();
                
            } catch (e) {
                console.error(e);
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'error',
                    message: '❌ Erreur lors de la suppression',
                    data: {}
                });
            }
        }
        
        async function clearAllIndexes() {
            if (!confirm('⚠️ ATTENTION: Ceci va supprimer TOUS les index RAG (documents_dense et documents_bm25).\\n\\nTous les documents ingérés seront perdus.\\n\\nContinuer ?')) {
                return;
            }
            
            try {
                const res = await fetch('/api/clear-all', { method: 'DELETE' });
                const data = await res.json();
                
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'success',
                    message: '✓ Tous les index RAG ont été supprimés',
                    data: data
                });
                
                await refreshIndexes();
                await loadStats();
                
            } catch (e) {
                console.error(e);
            }
        }
        
        // Export/Import functions
        async function exportSelectedIndex() {
            const indexName = document.getElementById('index-select').value;
            if (!indexName) {
                alert('Veuillez sélectionner un index à exporter');
                return;
            }
            
            addLog({
                timestamp: new Date().toISOString(),
                level: 'info',
                message: `📦 Export de l'index "${indexName}" en cours...`
            });
            
            try {
                const res = await fetch(`/api/index/${indexName}/export`, { method: 'POST' });
                const data = await res.json();
                
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'success',
                    message: `✓ Export terminé: ${data.qdrant || ''} | ${data.opensearch || ''}`,
                    data: data
                });
                
                alert(`Index sauvegardé`);
                
            } catch (e) {
                console.error(e);
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'error',
                    message: `❌ Erreur export: ${e.message}`
                });
            }
        }
        
        async function showImportDialog() {
            const section = document.getElementById('backups-section');
            section.style.display = 'block';
            
            // Charger la liste des backups
            try {
                const res = await fetch('/api/backups');
                const data = await res.json();
                
                const select = document.getElementById('backup-select');
                select.innerHTML = '<option value="">-- Sélectionner un backup --</option>';
                
                for (const backup of data.backups) {
                    const sizeKB = Math.round(backup.size / 1024);
                    const date = new Date(backup.modified * 1000).toLocaleString('fr-FR');
                    select.innerHTML += `<option value="${backup.name}">${backup.name} (${sizeKB} KB - ${date})</option>`;
                }
                
            } catch (e) {
                console.error(e);
            }
        }
        
        function hideImportDialog() {
            document.getElementById('backups-section').style.display = 'none';
        }
        
        async function importSelectedBackup() {
            const backupName = document.getElementById('backup-select').value;
            if (!backupName) {
                alert('Veuillez sélectionner un backup à importer');
                return;
            }
            
            if (!confirm(`Importer le backup "${backupName}" ?\\n\\nCeci ajoutera les données à l'index existant.`)) {
                return;
            }
            
            addLog({
                timestamp: new Date().toISOString(),
                level: 'info',
                message: `📥 Import de "${backupName}" en cours...`
            });
            
            try {
                const res = await fetch('/api/index/import', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ backup_file: backupName })
                });
                const data = await res.json();
                
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'success',
                    message: `✓ Import terminé: ${data.count} éléments importés`,
                    data: data
                });
                
                hideImportDialog();
                await refreshIndexes();
                await loadStats();
                
            } catch (e) {
                console.error(e);
                addLog({
                    timestamp: new Date().toISOString(),
                    level: 'error',
                    message: `❌ Erreur import: ${e.message}`
                });
            }
        }
        
        // Utils
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Les profils ont été supprimés - Configuration uniquement via settings.yaml
        
        function getCurrentParams() {
            // Les paramètres sont maintenant gérés uniquement via settings.yaml
            // On retourne null pour utiliser les valeurs par défaut du fichier YAML
            return null;
        }
        
        // Charger les paramètres depuis l'API (lecture du settings.yaml)
        async function loadConfigFromYaml() {
            try {
                const res = await fetch('/api/config');
                if (res.ok) {
                    const config = await res.json();
                    // Afficher les valeurs en lecture seule
                    if (config.chunking) {
                        document.getElementById('param-chunk-size').value = config.chunking.chunk_size || 512;
                        document.getElementById('param-overlap').value = config.chunking.chunk_overlap || 50;
                    }
                    if (config.retrieval) {
                        document.getElementById('param-bm25-topk').value = config.retrieval.bm25_top_k || 50;
                        document.getElementById('param-dense-topk').value = config.retrieval.dense_top_k || 50;
                        document.getElementById('param-fusion-topk').value = config.retrieval.fusion_top_k || 30;
                        document.getElementById('param-rerank-topn').value = config.retrieval.rerank_top_n || 10;
                    }
                    if (config.embedding) {
                        document.getElementById('param-batch-size').value = config.embedding.batch_size || 32;
                    }
                    addLog({
                        timestamp: new Date().toISOString(),
                        level: 'info',
                        message: '✓ Configuration chargée depuis settings.yaml',
                        data: {}
                    });
                }
            } catch (e) {
                console.log('Config API not available, using defaults');
            }
        }

        // ============================================================
        // Tab navigation
        // ============================================================
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(function(el) {
                el.classList.remove('active');
            });
            document.querySelectorAll('.nav-tab').forEach(function(el) {
                el.classList.remove('active');
            });
            document.getElementById('tab-' + tabName).classList.add('active');
            document.getElementById('nav-' + tabName).classList.add('active');

            if (tabName === 'ragas') {
                loadRagasData();
            }
        }

        // ============================================================
        // RAGAS tab
        // ============================================================
        var ragasRefreshInterval = null;
        var queryHistoryCache = [];

        async function clearRagasHistory() {
            if (!confirm('Effacer tout l\\'historique RAGAS (evaluations + requetes) ?\\nCette action est irreversible.')) return;
            try {
                var res = await fetch('/api/admin/ragas/clear', { method: 'DELETE' });
                if (res.ok) {
                    var data = await res.json();
                    alert('Historique efface : ' + data.deleted_scores + ' evaluations + ' + data.deleted_history + ' requetes supprimees.');
                    loadRagasData();
                    document.getElementById('ragas-history-body').innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-secondary)">Historique vide</td></tr>';
                } else {
                    alert('Erreur lors de la suppression.');
                }
            } catch(e) {
                console.error('clearRagasHistory error:', e);
                alert('Erreur : ' + e.message);
            }
        }

        async function loadRagasData() {
            try {
                var res = await fetch('/api/admin/ragas/summary');
                if (res.ok) {
                    var wrapper = await res.json();
                    var data = wrapper.summary || wrapper;
                    updateRagasCard('ragas-faithfulness', data.avg_faithfulness, 0.80);
                    updateRagasCard('ragas-relevancy', data.avg_answer_relevancy, 0.75);
                    updateRagasCard('ragas-recall', data.avg_context_recall, 0.70);
                }
            } catch (e) {
                console.log('Failed to load RAGAS summary:', e);
            }

            try {
                var res2 = await fetch('/api/admin/ragas/scores');
                if (res2.ok) {
                    var data2 = await res2.json();
                    var rows = data2.scores || [];
                    var tbody = document.getElementById('ragas-table-body');
                    if (rows.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">Aucune evaluation RAGAS pour le moment</td></tr>';
                    } else {
                        tbody.innerHTML = '';
                        rows.forEach(function(row) {
                            var tr = document.createElement('tr');
                            var date = (row.timestamp || '--').substring(0, 19).replace('T', ' ');
                            var question = row.question || '--';
                            if (question.length > 80) question = question.substring(0, 77) + '...';
                            tr.innerHTML = '<td>' + escapeHtml(date) + '</td>'
                                + '<td>' + escapeHtml(question) + '</td>'
                                + '<td class="' + ragasScoreClass(row.faithfulness, 0.80) + '">' + formatScore(row.faithfulness) + '</td>'
                                + '<td class="' + ragasScoreClass(row.answer_relevancy, 0.75) + '">' + formatScore(row.answer_relevancy) + '</td>'
                                + '<td class="' + ragasScoreClass(row.context_recall, 0.70) + '">' + formatScore(row.context_recall) + '</td>';
                            tbody.appendChild(tr);
                        });
                    }
                }
            } catch (e) {
                console.log('Failed to load RAGAS scores:', e);
            }

            if (!ragasRefreshInterval) {
                ragasRefreshInterval = setInterval(function() {
                    if (document.getElementById('tab-ragas').classList.contains('active')) {
                        loadRagasData();
                    }
                }, 60000);
            }
        }

        async function loadQueryHistory() {
            var tbody = document.getElementById('ragas-history-body');
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-secondary)">Chargement...</td></tr>';
            try {
                var res = await fetch('/api/admin/ragas/history');
                if (!res.ok) throw new Error('HTTP ' + res.status);
                var data = await res.json();
                var rows = data.queries || [];
                queryHistoryCache = rows;
                if (rows.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-secondary)">Aucune requete enregistree. Posez des questions dans le Chat pour alimenter l\\'historique.</td></tr>';
                    return;
                }
                tbody.innerHTML = '';
                rows.forEach(function(row, idx) {
                    var tr = document.createElement('tr');
                    var date = (row.timestamp || '--').substring(0, 19).replace('T', ' ');
                    var q = row.question || '--';
                    if (q.length > 60) q = q.substring(0, 57) + '...';
                    var answerLen = (row.answer || '').length;
                    var ctxCount = 0;
                    try { ctxCount = JSON.parse(row.contexts || '[]').length; } catch(e) {}
                    tr.innerHTML = '<td style="white-space:nowrap;font-size:0.8rem">' + escapeHtml(date) + '</td>'
                        + '<td title="' + escapeHtml(row.question || '') + '">' + escapeHtml(q) + '</td>'
                        + '<td style="text-align:center">' + (answerLen > 0 ? answerLen + ' car.' : '--') + '</td>'
                        + '<td style="text-align:center">' + ctxCount + ' docs</td>'
                        + '<td><button class="btn btn-primary" style="padding:4px 12px;font-size:0.8rem" '
                        + 'onclick="evaluateQuery(' + idx + ')" id="eval-btn-' + idx + '"'
                        + (ctxCount === 0 ? ' disabled title="Pas de contextes disponibles"' : '')
                        + '>Evaluer</button></td>';
                    tbody.appendChild(tr);
                });
            } catch (e) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--error)">Erreur: ' + escapeHtml(e.message) + '</td></tr>';
            }
        }

        async function evaluateQuery(idx) {
            var row = queryHistoryCache[idx];
            if (!row) return;
            var btn = document.getElementById('eval-btn-' + idx);
            if (btn) { btn.textContent = '...'; btn.disabled = true; }
            try {
                var contexts = [];
                try { contexts = JSON.parse(row.contexts || '[]'); } catch(e) {}
                var res = await fetch('/api/admin/ragas/evaluate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: row.question,
                        answer: row.answer || '',
                        contexts: contexts,
                        index_name: row.index_name || 'documents'
                    })
                });
                if (!res.ok) {
                    var err = await res.text();
                    throw new Error(err);
                }
                var data = await res.json();
                if (btn) {
                    var s = data.scores || {};
                    btn.textContent = 'F:' + formatScore(s.faithfulness) + ' R:' + formatScore(s.answer_relevancy) + ' C:' + formatScore(s.context_recall);
                    btn.style.fontSize = '0.7rem';
                    btn.style.background = '#1a3a1a';
                    btn.style.borderColor = 'var(--success)';
                }
                loadRagasData();
            } catch (e) {
                if (btn) { btn.textContent = 'Erreur'; btn.style.background = '#3a1a1a'; }
                console.error('RAGAS eval error:', e);
            }
        }

        function updateRagasCard(elementId, value, threshold) {
            var el = document.getElementById(elementId);
            if (value === null || value === undefined) {
                el.textContent = '--';
                el.className = 'ragas-card-value';
            } else {
                el.textContent = value.toFixed(3);
                el.className = 'ragas-card-value ' + (value >= threshold ? 'good' : 'bad');
            }
        }

        function ragasScoreClass(value, threshold) {
            if (value === null || value === undefined) return '';
            return value >= threshold ? 'ragas-score-good' : 'ragas-score-bad';
        }

        function formatScore(value) {
            if (value === null || value === undefined) return '--';
            return value.toFixed(3);
        }

        // ============================================================
        // RAGAS Batch Evaluation
        // ============================================================

        async function loadRagasDatasets() {
            var container = document.getElementById('ragas-datasets-list');
            container.innerHTML = '<em style="color:var(--text-secondary)">Chargement...</em>';
            try {
                var res = await fetch('/api/admin/ragas/datasets');
                var data = await res.json();
                var datasets = data.datasets || [];
                if (datasets.length === 0) {
                    container.innerHTML = '<em style="color:var(--text-secondary)">Aucun dataset trouve dans evaluation/datasets/. Placez un fichier JSON au format {name, pairs: [{question, ground_truth}]}.</em>';
                    return;
                }
                var html = '<div style="display:flex;flex-direction:column;gap:10px">';
                datasets.forEach(function(ds) {
                    html += '<div style="display:flex;align-items:center;justify-content:space-between;padding:12px;background:var(--bg-elevated);border-radius:8px;border:1px solid var(--border)">'
                        + '<div>'
                        + '<strong style="color:var(--text-primary)">' + escapeHtml(ds.name) + '</strong>'
                        + '<div style="color:var(--text-secondary);font-size:0.8rem">' + escapeHtml(ds.description) + '</div>'
                        + '<div style="color:var(--accent);font-size:0.8rem">' + ds.pairs_count + ' paires Q/R</div>'
                        + '</div>'
                        + '<button class="btn btn-primary" onclick="runBatchEval(\\'' + escapeHtml(ds.path) + '\\', \\'' + escapeHtml(ds.name) + '\\')" '
                        + 'id="batch-btn-' + escapeHtml(ds.filename) + '"'
                        + '>Lancer RAGAS</button>'
                        + '</div>';
                });
                html += '</div>';
                container.innerHTML = html;
            } catch (e) {
                container.innerHTML = '<em style="color:var(--error)">Erreur: ' + escapeHtml(e.message) + '</em>';
            }
        }

        function renderAnalysis(analysis) {
            if (!analysis || !analysis.recommendations) return '';
            var hasProblems = analysis.recommendations.some(function(r) { return r.severity !== 'info'; });
            var headerClass = hasProblems ? (analysis.recommendations.some(function(r) { return r.severity === 'high'; }) ? 'bad' : 'warn') : 'ok';
            var icon = headerClass === 'ok' ? '&#10003;' : (headerClass === 'bad' ? '&#9888;' : '&#9888;');

            var html = '<div class="analysis-panel">';
            html += '<div class="analysis-header ' + headerClass + '">';
            html += '<span style="font-size:1.3rem">' + icon + '</span>';
            html += '<h3>Diagnostic &amp; Recommandations</h3>';
            html += '<span style="margin-left:auto;font-size:0.85rem;color:var(--text-secondary)">' + escapeHtml(analysis.summary) + '</span>';
            html += '</div>';

            var avg = analysis.scores_avg || {};
            var thr = analysis.thresholds || {};
            if (avg.faithfulness != null || avg.answer_relevancy != null || avg.context_recall != null) {
                html += '<div class="analysis-averages">';
                var metrics = [
                    {key: 'faithfulness', label: 'Fidelite', th: thr.faithfulness || 0.8},
                    {key: 'answer_relevancy', label: 'Pertinence', th: thr.answer_relevancy || 0.75},
                    {key: 'context_recall', label: 'Rappel ctx', th: thr.context_recall || 0.7}
                ];
                metrics.forEach(function(m) {
                    var val = avg[m.key];
                    var cls = val == null ? '' : (val >= m.th ? 'good' : (val >= m.th * 0.75 ? 'warn' : 'bad'));
                    html += '<div class="analysis-avg-item">';
                    html += '<span class="analysis-avg-label">' + m.label + '</span>';
                    html += '<span class="analysis-avg-value ' + cls + '">' + (val != null ? (val * 100).toFixed(1) + '%' : 'N/A') + '</span>';
                    html += '<span class="analysis-avg-threshold">seuil : ' + (m.th * 100).toFixed(0) + '%</span>';
                    html += '</div>';
                });
                html += '</div>';
            }

            html += '<div class="analysis-recs">';
            analysis.recommendations.forEach(function(rec) {
                html += '<div class="analysis-rec">';
                html += '<div class="analysis-rec-header">';
                html += '<span class="analysis-severity ' + rec.severity + '">' + rec.severity + '</span>';
                html += '<span class="analysis-rec-metric">' + escapeHtml(rec.metric) + '</span>';
                if (rec.score_avg != null) {
                    html += '<span class="analysis-rec-score">moyenne ' + (rec.score_avg * 100).toFixed(1) + '% &mdash; ' + (rec.failing_pct || 0) + '% sous le seuil</span>';
                }
                html += '</div>';
                html += '<div class="analysis-diagnostic">' + escapeHtml(rec.diagnostic) + '</div>';
                if (rec.param_changes && rec.param_changes.length > 0) {
                    html += '<table class="ragas-table" style="margin:10px 0;font-size:0.85rem"><thead><tr>'
                        + '<th>Parametre</th><th>Fichier</th><th>Actuel</th><th style="color:var(--accent)">Suggere</th><th>Raison</th></tr></thead><tbody>';
                    rec.param_changes.forEach(function(p) {
                        var changed = String(p.current) !== String(p.suggested);
                        html += '<tr>'
                            + '<td style="font-family:monospace;font-weight:600">' + escapeHtml(p.param) + '</td>'
                            + '<td style="color:var(--text-secondary)">' + escapeHtml(p.file) + '</td>'
                            + '<td>' + escapeHtml(String(p.current)) + '</td>'
                            + '<td style="color:' + (changed ? 'var(--accent);font-weight:700' : 'var(--text-secondary)') + '">' + escapeHtml(String(p.suggested)) + (changed ? ' &#x2190;' : '') + '</td>'
                            + '<td style="color:var(--text-secondary);font-size:0.8rem">' + escapeHtml(p.reason) + '</td></tr>';
                    });
                    html += '</tbody></table>';
                }
                if (rec.actions && rec.actions.length > 0) {
                    html += '<ul class="analysis-actions">';
                    rec.actions.forEach(function(a) {
                        html += '<li>' + escapeHtml(a) + '</li>';
                    });
                    html += '</ul>';
                }
                html += '</div>';
            });
            html += '</div></div>';
            return html;
        }

        async function simulateRagas() {
            var btn = document.getElementById('btn-simulate');
            var statusDiv = document.getElementById('feedback-eval-status');
            var progressDiv = document.getElementById('feedback-eval-progress');
            var resultsDiv = document.getElementById('feedback-eval-results');

            btn.disabled = true;
            statusDiv.style.display = 'block';
            progressDiv.innerHTML = '<strong>Simulation en cours...</strong>';
            resultsDiv.innerHTML = '';

            try {
                var res = await fetch('/api/admin/ragas/simulate', { method: 'POST' });
                var data = await res.json();
                btn.disabled = false;

                progressDiv.innerHTML = '<strong style="color:var(--accent)">SIMULATION (scores fictifs)</strong> &mdash; '
                    + data.evaluated + ' feedbacks simules pour demontrer le systeme de diagnostic.';

                var html = '';
                if (data.analysis) { html += renderAnalysis(data.analysis); }

                html += '<table class="ragas-table" style="margin-top:10px"><thead><tr>'
                    + '<th>FB#</th><th>Question</th><th>Faith.</th><th>Relev.</th><th>Recall</th>'
                    + '</tr></thead><tbody>';
                data.results.forEach(function(r) {
                    var s = r.scores;
                    html += '<tr><td>' + r.feedback_id + '</td>'
                        + '<td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeHtml(r.question) + '</td>'
                        + '<td class="' + ragasScoreClass(s.faithfulness, 0.80) + '">' + formatScore(s.faithfulness) + '</td>'
                        + '<td class="' + ragasScoreClass(s.answer_relevancy, 0.75) + '">' + formatScore(s.answer_relevancy) + '</td>'
                        + '<td class="' + ragasScoreClass(s.context_recall, 0.70) + '">' + formatScore(s.context_recall) + '</td></tr>';
                });
                html += '</tbody></table>';
                resultsDiv.innerHTML = html;
            } catch (e) {
                btn.disabled = false;
                progressDiv.innerHTML = '<strong style="color:var(--error)">Erreur: ' + escapeHtml(e.message) + '</strong>';
            }
        }

        async function evaluateFeedbacks() {
            var btn = document.getElementById('btn-eval-feedbacks');
            var statusDiv = document.getElementById('feedback-eval-status');
            var progressDiv = document.getElementById('feedback-eval-progress');
            var resultsDiv = document.getElementById('feedback-eval-results');

            btn.disabled = true;
            btn.textContent = 'Evaluation en cours...';
            statusDiv.style.display = 'block';
            progressDiv.innerHTML = '<strong>Evaluation RAGAS des feedbacks en cours...</strong><br>'
                + 'Pour chaque feedback negatif avec correction, le RAG est re-interroge puis RAGAS compare avec la reponse attendue. Cela peut prendre plusieurs minutes.';
            resultsDiv.innerHTML = '';

            try {
                var res = await fetch('/api/admin/ragas/evaluate-feedbacks', { method: 'POST' });
                var data = await res.json();

                btn.disabled = false;
                btn.textContent = 'Lancer l\\'evaluation RAGAS';

                if (data.total === 0) {
                    progressDiv.innerHTML = '<strong style="color:var(--text-secondary)">Aucun feedback negatif avec correction a evaluer.</strong>';
                    return;
                }

                if (data.status === 'completed') {
                    progressDiv.innerHTML = '<strong style="color:var(--success)">Evaluation terminee !</strong> '
                        + data.evaluated + '/' + data.total + ' feedbacks evalues.';

                    var html = '';

                    if (data.analysis) {
                        html += renderAnalysis(data.analysis);
                    }

                    html += '<table class="ragas-table" style="margin-top:10px"><thead><tr>'
                        + '<th>FB#</th><th>Question</th><th>Correction (ground truth)</th><th>Faith.</th><th>Relev.</th><th>Recall</th><th>Statut</th>'
                        + '</tr></thead><tbody>';
                    data.results.forEach(function(r) {
                        if (r.status === 'ok') {
                            var s = r.scores;
                            html += '<tr>'
                                + '<td>' + r.feedback_id + '</td>'
                                + '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + escapeHtml(r.question) + '">' + escapeHtml(r.question) + '</td>'
                                + '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + escapeHtml(r.ground_truth || '') + '">' + escapeHtml(r.ground_truth || '') + '</td>'
                                + '<td class="' + ragasScoreClass(s.faithfulness, 0.80) + '">' + formatScore(s.faithfulness) + '</td>'
                                + '<td class="' + ragasScoreClass(s.answer_relevancy, 0.75) + '">' + formatScore(s.answer_relevancy) + '</td>'
                                + '<td class="' + ragasScoreClass(s.context_recall, 0.70) + '">' + formatScore(s.context_recall) + '</td>'
                                + '<td style="color:var(--success)">OK</td></tr>';
                        } else {
                            html += '<tr><td>' + (r.feedback_id || '-') + '</td>'
                                + '<td>' + escapeHtml(r.question || '') + '</td>'
                                + '<td>-</td>'
                                + '<td colspan="3" style="color:var(--error)">' + escapeHtml(r.reason || r.status) + '</td>'
                                + '<td style="color:var(--error)">' + r.status + '</td></tr>';
                        }
                    });
                    html += '</tbody></table>';
                    resultsDiv.innerHTML = html;
                    loadRagasData();
                } else {
                    progressDiv.innerHTML = '<strong style="color:var(--error)">Erreur: ' + escapeHtml(data.detail || JSON.stringify(data)) + '</strong>';
                }
            } catch (e) {
                btn.disabled = false;
                btn.textContent = 'Lancer l\\'evaluation RAGAS';
                progressDiv.innerHTML = '<strong style="color:var(--error)">Erreur: ' + escapeHtml(e.message) + '</strong>';
            }
        }

        async function runBatchEval(datasetPath, datasetName) {
            var statusDiv = document.getElementById('ragas-batch-status');
            var progressDiv = document.getElementById('ragas-batch-progress');
            var resultsDiv = document.getElementById('ragas-batch-results');

            statusDiv.style.display = 'block';
            progressDiv.innerHTML = '<strong>Evaluation en cours : ' + escapeHtml(datasetName) + '</strong><br>'
                + 'Pour chaque question, le RAG est interroge puis RAGAS evalue les metriques. Cela peut prendre plusieurs minutes...';
            resultsDiv.innerHTML = '';

            try {
                var res = await fetch('/api/admin/ragas/batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({dataset_path: datasetPath, index_name: 'documents'})
                });
                var data = await res.json();

                if (data.status === 'completed') {
                    progressDiv.innerHTML = '<strong style="color:var(--success)">Evaluation terminee !</strong> '
                        + data.evaluated + '/' + data.total + ' questions evaluees.';

                    var html = '<table class="ragas-table" style="margin-top:10px"><thead><tr>'
                        + '<th>#</th><th>Question</th><th>Faithfulness</th><th>Relevancy</th><th>Ctx Recall</th><th>Statut</th>'
                        + '</tr></thead><tbody>';
                    data.results.forEach(function(r, i) {
                        if (r.status === 'ok') {
                            var s = r.scores;
                            html += '<tr>'
                                + '<td>' + (i+1) + '</td>'
                                + '<td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + escapeHtml(r.question) + '">' + escapeHtml(r.question) + '</td>'
                                + '<td class="' + ragasScoreClass(s.faithfulness, 0.80) + '">' + formatScore(s.faithfulness) + '</td>'
                                + '<td class="' + ragasScoreClass(s.answer_relevancy, 0.75) + '">' + formatScore(s.answer_relevancy) + '</td>'
                                + '<td class="' + ragasScoreClass(s.context_recall, 0.70) + '">' + formatScore(s.context_recall) + '</td>'
                                + '<td style="color:var(--success)">OK</td></tr>';
                        } else {
                            html += '<tr><td>' + (i+1) + '</td>'
                                + '<td>' + escapeHtml(r.question || '') + '</td>'
                                + '<td colspan="3" style="color:var(--error)">' + escapeHtml(r.reason || r.status) + '</td>'
                                + '<td style="color:var(--error)">' + r.status + '</td></tr>';
                        }
                    });
                    html += '</tbody></table>';
                    resultsDiv.innerHTML = html;
                    loadRagasData();
                } else {
                    progressDiv.innerHTML = '<strong style="color:var(--error)">Erreur: ' + escapeHtml(data.detail || JSON.stringify(data)) + '</strong>';
                }
            } catch (e) {
                progressDiv.innerHTML = '<strong style="color:var(--error)">Erreur: ' + escapeHtml(e.message) + '</strong>';
            }
        }

    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Luciole RAG - Interface d'Ingestion")
    print("  http://localhost:8002")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080)
