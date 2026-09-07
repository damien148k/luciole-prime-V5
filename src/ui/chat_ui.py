"""
Chat UI - Interface de conversation avec l'Agent Luciole
V3 Constellation: refonte UI inspirée Claude/ChatGPT/Perplexity
- Welcome screen avec logo + abdomen scintillant
- Conversation style Perplexity (question grand titre, réponse, citations cliquables)
- Sidebar droite Sources/Passages
- Drawer paramètres (right slide-in)
- Logique conservée: history, settings localStorage, feedback key users, query rewriting
"""

import os
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
from loguru import logger

app = FastAPI(
    title="Luciole Chat",
    description="Interface de conversation avec l'Agent Luciole",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# V4 : plus de conteneur agent. Le chat parle directement au coeur RAG
# (service rag) par son contrat public POST /api/rag/query. RAG_URL
# remplace AGENT_URL de la V3 ; l'ancien nom reste accepte pour ne pas
# casser un .env existant.
RAG_URL = os.environ.get("RAG_URL") or os.environ.get("AGENT_URL", "http://rag:8000")
SERVICE_NAME = os.environ.get("INSTANCE_NAME", "Luciole")

# Service de feedback optionnel (couche 3, deployee a part) : sans lui,
# le chat masque simplement le bouton d'avis au lieu d'echouer.
FEEDBACK_URL = os.environ.get("FEEDBACK_URL", "")

# Le chat interroge le pipeline iteratif du coeur RAG : recherche,
# analyse de couverture, recherche ciblee optionnelle, puis generation.
# Il n'emploie ni boucle agentique ni reecriture de requete.

# Chemin vers les assets statiques (logo + polices offline)
STATIC_DIR = Path(__file__).parent / "static"
LOGO_PATH = STATIC_DIR / "logo.png"

# Sert /static/fonts/* (woff2) et tout autre asset du dossier static/
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/logo.png")
async def get_logo():
    """Servir le logo Luciole"""
    if LOGO_PATH.exists():
        return FileResponse(LOGO_PATH, media_type="image/png")
    # Fallback: chercher dans pics/
    alt_path = Path(__file__).parent.parent.parent / "pics" / "luciole-logo.png"
    if alt_path.exists():
        return FileResponse(alt_path, media_type="image/png")
    return HTMLResponse("Logo not found", status_code=404)


@app.get("/favicon.ico")
@app.get("/favicon.png")
async def get_favicon():
    """Servir le favicon dedie (tuile indigo lisible sur onglet clair).

    Avant, on servait logo.png : luciole blanche sur fond transparent,
    invisible dans les onglets en theme clair. favicon.png a desormais son
    propre visuel ; logo.png reste utilise dans les pages (fond sombre).
    """
    favicon_path = STATIC_DIR / "favicon.png"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/png")
    if LOGO_PATH.exists():
        return FileResponse(LOGO_PATH, media_type="image/png")
    alt_path = Path(__file__).parent.parent.parent / "pics" / "luciole-logo.png"
    if alt_path.exists():
        return FileResponse(alt_path, media_type="image/png")
    return HTMLResponse("Favicon not found", status_code=404)


class ChatMessage(BaseModel):
    role: str  # "user" ou "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    index_name: Optional[str] = None
    top_k: int = 20
    custom_prompt: Optional[str] = None  # Prompt personnalisé optionnel
    # Recherche approfondie : profil elargi du pipeline iteratif
    # (reglages query2.deep de settings.yaml, PR #48). Quand le toggle
    # est actif, le front lance AUSSI une seconde passe deep, affichee
    # repliee en « Pistes complementaires » sous la reponse standard.
    deep_search: bool = False
    history: list[ChatMessage] = []  # Historique de conversation


@app.get("/", response_class=HTMLResponse)
async def chat_page():
    """Page de chat principale"""
    # Titre dynamique basé sur le métier
    service_suffix = f" {SERVICE_NAME}" if SERVICE_NAME else ""
    page_title = f"Luciole Chat{service_suffix}"

    html = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{PAGE_TITLE}}</title>
<link rel="icon" type="image/png" href="/favicon.png">
<link rel="shortcut icon" type="image/png" href="/favicon.png">
<!-- Polices servies localement (offline) -->
<link rel="stylesheet" href="/static/fonts/fonts.css">
<style>
:root{
  --bg-deep:#060a18; --bg-primary:#0b1228; --bg-secondary:#111a36;
  --bg-tertiary:#1a2447; --bg-elevated:#1f2a55;
  --gold:#f5c97a; --gold-bright:#ffd98a; --gold-soft:#e8b964; --gold-deep:#b8862d;
  --gold-glow:rgba(245,201,122,0.35); --gold-glow-soft:rgba(245,201,122,0.12);
  --text-primary:#eef2ff; --text-secondary:#a8b2d1; --text-muted:#6b7595;
  --border:rgba(168,178,209,0.10); --border-strong:rgba(245,201,122,0.25);
  --success:#5eead4; --error:#f87171;
  --radius-sm:8px; --radius:14px; --radius-lg:20px; --radius-xl:28px;
  --shadow-glow:0 0 24px var(--gold-glow);
  --shadow-card:0 8px 32px rgba(0,0,0,0.4);
  --shadow-soft:0 2px 12px rgba(0,0,0,0.25);
}

*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:var(--bg-deep);
  color:var(--text-primary);
  font-size:15px;
  line-height:1.6;
  overflow:hidden;
  -webkit-font-smoothing:antialiased;
}

/* === Fond constellation === */
body::before{
  content:'';
  position:fixed;inset:0;
  background:
    radial-gradient(ellipse at 18% 12%, rgba(245,201,122,0.08) 0%, transparent 45%),
    radial-gradient(ellipse at 88% 78%, rgba(120,150,255,0.05) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 50%, rgba(245,201,122,0.025) 0%, transparent 70%),
    linear-gradient(180deg,#060a18 0%,#0b1228 50%,#060a18 100%);
  z-index:-2;
}
body::after{
  content:'';
  position:fixed;inset:0;
  background-image:
    radial-gradient(1.5px 1.5px at 20% 30%, rgba(245,201,122,0.55), transparent),
    radial-gradient(1px 1px at 60% 70%, rgba(245,201,122,0.45), transparent),
    radial-gradient(1.5px 1.5px at 80% 20%, rgba(245,201,122,0.35), transparent),
    radial-gradient(1px 1px at 35% 85%, rgba(255,217,138,0.45), transparent),
    radial-gradient(1.5px 1.5px at 90% 55%, rgba(245,201,122,0.25), transparent),
    radial-gradient(1px 1px at 10% 60%, rgba(255,217,138,0.35), transparent),
    radial-gradient(1px 1px at 45% 15%, rgba(245,201,122,0.4), transparent),
    radial-gradient(1.5px 1.5px at 70% 45%, rgba(255,217,138,0.3), transparent);
  animation:twinkle 9s ease-in-out infinite;
  z-index:-1;
}
@keyframes twinkle{0%,100%{opacity:.4}50%{opacity:.95}}
@keyframes glow{0%,100%{opacity:.65;transform:scale(1)}50%{opacity:1;transform:scale(1.08)}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}

/* === Layout === */
.app{display:flex;flex-direction:column;height:100vh}
.body-row{display:flex;flex:1;min-height:0;position:relative}
.body-row .main{flex:1;min-width:0}

/* Sidebar droite : sources + passages */
.sources-sidebar{
  width:380px;flex-shrink:0;
  background:rgba(11,18,40,0.7);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border-left:1px solid var(--border);
  display:none;flex-direction:column;
  overflow:hidden;
  animation:slideInRight .3s cubic-bezier(.4,0,.2,1);
}
.sources-sidebar.open{display:flex}
@keyframes slideInRight{from{transform:translateX(20px);opacity:0}to{transform:translateX(0);opacity:1}}

.sb-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 20px;border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.sb-title{
  display:flex;align-items:center;gap:10px;
  font-family:'Cormorant Garamond',serif;font-style:italic;font-weight:600;
  font-size:1.25rem;
  background:linear-gradient(135deg,var(--gold-bright),var(--gold));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.sb-title svg{width:16px;height:16px;color:var(--gold)}
.sb-count{
  font-family:'Inter',sans-serif;font-style:normal;
  font-size:.72rem;font-weight:600;
  padding:2px 8px;background:var(--gold-glow-soft);
  border:1px solid var(--border-strong);border-radius:999px;
  color:var(--gold);
}
.sb-tabs{
  display:flex;gap:4px;padding:10px 20px 0;
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.sb-tab{
  padding:8px 14px;background:transparent;border:none;
  font-size:.82rem;font-weight:500;color:var(--text-muted);
  cursor:pointer;border-bottom:2px solid transparent;
  transition:all .15s;margin-bottom:-1px;
}
.sb-tab:hover{color:var(--text-secondary)}
.sb-tab.active{color:var(--gold);border-bottom-color:var(--gold)}
.sb-body{flex:1;overflow-y:auto;padding:16px 20px}
.sb-body::-webkit-scrollbar{width:6px}
.sb-body::-webkit-scrollbar-thumb{background:var(--bg-tertiary);border-radius:3px}
.sb-pane{display:none}
.sb-pane.active{display:block;animation:fadeInUp .25s ease-out}

.sb-source{
  display:flex;align-items:flex-start;gap:10px;
  padding:12px;margin-bottom:8px;
  background:rgba(17,26,54,0.5);
  border:1px solid var(--border);border-radius:10px;
  cursor:pointer;transition:all .2s;
  text-decoration:none;color:inherit;
}
.sb-source:hover{border-color:var(--border-strong);background:rgba(26,36,71,0.7);transform:translateX(-2px)}
.sb-source-num{
  width:22px;height:22px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  background:var(--gold-glow-soft);border:1px solid var(--border-strong);
  border-radius:6px;
  font-size:.72rem;font-weight:700;color:var(--gold);
}
.sb-source-body{flex:1;min-width:0}
.sb-source-title{
  font-size:.86rem;font-weight:600;color:var(--text-primary);
  line-height:1.4;margin-bottom:4px;
  word-break:break-word;
}
.sb-source-meta{
  display:flex;align-items:center;gap:6px;
  font-size:.72rem;color:var(--text-muted);
}
.sb-source-score{
  margin-left:auto;font-family:'JetBrains Mono',monospace;
  font-size:.7rem;color:var(--gold);
}

.sb-passage{
  padding:12px 14px;margin-bottom:10px;
  background:var(--bg-secondary);
  border-left:3px solid var(--gold);
  border-radius:0 10px 10px 0;
  font-size:.85rem;line-height:1.6;color:var(--text-primary);
  word-break:break-word;
}
.sb-passage-meta{
  display:flex;align-items:center;gap:8px;
  font-size:.7rem;color:var(--text-muted);margin-bottom:8px;
  text-transform:uppercase;letter-spacing:1px;font-weight:600;
}
.sb-passage-num{
  width:18px;height:18px;display:flex;align-items:center;justify-content:center;
  background:var(--gold);color:#1a1208;border-radius:50%;
  font-size:.66rem;font-weight:700;
}
.sb-empty{
  text-align:center;padding:40px 20px;color:var(--text-muted);
  font-size:.88rem;
}

/* Bouton flottant pour réouvrir la sidebar */
.sb-toggle{
  position:fixed;top:80px;right:20px;
  width:44px;height:44px;
  display:none;align-items:center;justify-content:center;
  background:rgba(17,26,54,0.9);
  backdrop-filter:blur(12px);
  border:1px solid var(--border-strong);border-radius:12px;
  color:var(--gold);cursor:pointer;
  z-index:30;transition:all .2s;
  box-shadow:var(--shadow-card);
}
.sb-toggle:hover{background:var(--gold-glow-soft);transform:translateY(-1px)}
.sb-toggle.show{display:flex}
.sb-toggle svg{width:20px;height:20px}
.sb-toggle .badge{
  position:absolute;top:-4px;right:-4px;
  min-width:18px;height:18px;padding:0 5px;
  background:var(--gold);color:#1a1208;
  font-family:'Inter',sans-serif;font-size:.68rem;font-weight:700;
  border-radius:9px;display:flex;align-items:center;justify-content:center;
  border:2px solid var(--bg-deep);
}

/* === Header === */
.header{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 28px;
  border-bottom:1px solid var(--border);
  background:rgba(11,18,40,0.6);
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  z-index:10;
  flex-shrink:0;
}
.brand{display:flex;align-items:center;gap:12px}
.brand-logo-wrap{position:relative;width:36px;height:36px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.brand-logo-wrap::before{
  content:'';position:absolute;inset:-3px;
  background:radial-gradient(circle,var(--gold-glow) 0%,transparent 65%);
  border-radius:50%;filter:blur(8px);
  animation:glow 4s ease-in-out infinite;z-index:0;
}
.brand-logo{width:100%;height:100%;object-fit:contain;position:relative;z-index:1;filter:sepia(1) saturate(4) hue-rotate(5deg) brightness(.97) drop-shadow(0 0 6px rgba(255,220,140,.7))}
.brand-name{
  font-family:'Cormorant Garamond',serif;
  font-style:italic;font-weight:600;font-size:1.45rem;
  background:linear-gradient(135deg,var(--gold-bright) 0%,var(--gold) 50%,var(--gold-soft) 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  letter-spacing:.5px;
}
.brand-tag{font-size:.72rem;color:var(--text-muted);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px}

.header-center{display:flex;align-items:center;gap:8px}
.index-pill{
  display:flex;align-items:center;gap:8px;
  padding:7px 14px;background:var(--bg-secondary);
  border:1px solid var(--border);border-radius:999px;
  font-size:.85rem;color:var(--text-secondary);
  cursor:pointer;transition:all .2s;
}
.index-pill:hover{border-color:var(--border-strong);color:var(--text-primary)}
.index-pill .dot{width:6px;height:6px;border-radius:50%;background:var(--success);box-shadow:0 0 8px var(--success)}
.index-pill .dot.error{background:var(--error);box-shadow:0 0 8px var(--error)}
.index-pill svg{width:12px;height:12px;opacity:.6}

.header-actions{display:flex;align-items:center;gap:6px}
.user-badge{
  font-size:.78rem;color:var(--text-muted);padding:6px 10px;
  border:1px solid var(--border);border-radius:8px;cursor:pointer;
  transition:all .2s;background:transparent;
}
.user-badge:hover{border-color:var(--border-strong);color:var(--gold)}
.icon-btn{
  width:36px;height:36px;display:flex;align-items:center;justify-content:center;
  background:transparent;border:1px solid transparent;border-radius:10px;
  color:var(--text-secondary);cursor:pointer;transition:all .2s;
}
.icon-btn:hover{background:var(--bg-secondary);border-color:var(--border);color:var(--gold)}
.icon-btn svg{width:18px;height:18px}
.btn-new{
  display:flex;align-items:center;gap:6px;
  padding:5px 11px;
  background:linear-gradient(135deg,var(--gold) 0%,var(--gold-soft) 100%);
  color:#1a1208;border:none;border-radius:8px;
  font-size:.8rem;font-weight:600;cursor:pointer;
  transition:all .2s;box-shadow:0 0 0 0 var(--gold-glow);
}
.btn-new:hover{transform:translateY(-1px);box-shadow:0 4px 16px var(--gold-glow)}
.btn-new svg{width:14px;height:14px}

/* Hidden native select that drives the index pill text */
#index{
  position:absolute;opacity:0;pointer-events:none;width:0;height:0;
}

/* === Main scroll area === */
.main{
  flex:1;overflow-y:auto;
  scroll-behavior:smooth;
  position:relative;
}
.main::-webkit-scrollbar{width:8px}
.main::-webkit-scrollbar-thumb{background:var(--bg-tertiary);border-radius:4px}
.main::-webkit-scrollbar-thumb:hover{background:var(--bg-elevated)}

/* === ÉCRAN D'ACCUEIL === */
.welcome{
  min-height:100%;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:40px 24px;
  animation:fadeInUp .6s ease-out;
}
.welcome-logo{
  position:relative;width:96px;height:96px;
  display:flex;align-items:center;justify-content:center;
  margin-bottom:32px;
}
.welcome-logo::before{
  content:'';position:absolute;inset:-20px;
  background:radial-gradient(circle,var(--gold-glow) 0%,transparent 70%);
  border-radius:50%;filter:blur(20px);
  animation:glow 4s ease-in-out infinite;
}
.welcome-logo img{width:100%;height:100%;object-fit:contain;position:relative;z-index:1;filter:sepia(1) saturate(4) hue-rotate(5deg) brightness(.97) drop-shadow(0 0 16px rgba(255,220,140,.8))}

/* Abdomen scintillant : 5 petits points lumineux */
.welcome-logo .abdomen{
  position:absolute;
  left:48%;top:68%;
  width:18px;height:18px;
  transform:translate(-50%,-50%);
  z-index:2;
  pointer-events:none;
}
.welcome-logo .abdomen::before{
  content:'';position:absolute;inset:-18px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(255,220,140,0.45) 0%,rgba(245,201,122,0.18) 45%,transparent 75%);
  filter:blur(8px);
  animation:abdomenHalo 3.2s ease-in-out infinite;
}
.welcome-logo .spark{
  position:absolute;
  width:2px;height:2px;
  border-radius:50%;
  background:radial-gradient(circle,#fff8e0 0%,var(--gold-bright) 50%,var(--gold) 100%);
  box-shadow:0 0 3px rgba(255,220,140,0.8);
  animation:sparkle 3.2s ease-in-out infinite;
}
.welcome-logo .spark.s1{left:0px;top:9px;animation-delay:0s}
.welcome-logo .spark.s2{left:5px;top:3px;animation-delay:.6s}
.welcome-logo .spark.s3{left:6px;top:11px;animation-delay:1.2s}
.welcome-logo .spark.s4{left:2px;top:5px;animation-delay:1.8s}
.welcome-logo .spark.s5{left:8px;top:7px;animation-delay:2.4s}

.welcome-logo.thinking-mode .abdomen::before{animation:abdomenHaloFast 0.9s ease-in-out infinite}
.welcome-logo.thinking-mode .spark{animation:sparkleFast 0.7s ease-in-out infinite}
.welcome-logo.thinking-mode .spark.s2{animation-delay:.12s}
.welcome-logo.thinking-mode .spark.s3{animation-delay:.24s}
.welcome-logo.thinking-mode .spark.s4{animation-delay:.36s}
.welcome-logo.thinking-mode .spark.s5{animation-delay:.48s}

@keyframes sparkle{
  0%,100%{opacity:.25;transform:scale(.6)}
  50%{opacity:1;transform:scale(1.4)}
}
@keyframes sparkleFast{
  0%,100%{opacity:.4;transform:scale(.8)}
  50%{opacity:1;transform:scale(1.6)}
}
@keyframes abdomenHalo{
  0%,100%{opacity:.4;transform:scale(.9)}
  50%{opacity:.95;transform:scale(1.2)}
}
@keyframes abdomenHaloFast{
  0%,100%{opacity:.55;transform:scale(1)}
  50%{opacity:1;transform:scale(1.4)}
}

.welcome-composer{
  width:100%;max-width:760px;margin-bottom:32px;
}

/* === ZONE DE CONVERSATION === */
.conversation{
  max-width:780px;margin:0 auto;
  padding:32px 24px 200px;
}

.user-block{
  margin-bottom:32px;
  animation:fadeInUp .4s ease-out;
}
.user-question{
  font-family:'Cormorant Garamond',serif;
  font-style:italic;font-weight:600;font-size:1.85rem;
  line-height:1.3;color:var(--text-primary);
  letter-spacing:.2px;
  word-wrap:break-word;
}

.answer-block{
  margin-bottom:48px;
  animation:fadeInUp .5s ease-out;
}

.sources-section{margin-bottom:24px}
.sources-label{
  display:flex;align-items:center;gap:8px;
  font-size:.78rem;font-weight:600;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:1.5px;
  margin-bottom:12px;
}
.sources-label svg{width:14px;height:14px;color:var(--gold)}
.sources-grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;
}
.source-card{
  position:relative;
  padding:12px 12px 12px 14px;
  background:rgba(17,26,54,0.5);
  border:1px solid var(--border);border-radius:12px;
  cursor:pointer;transition:all .2s;
  text-decoration:none;color:inherit;
  overflow:hidden;
}
.source-card:hover{
  border-color:var(--border-strong);
  background:rgba(26,36,71,0.7);
  transform:translateY(-1px);
}
.source-num{
  position:absolute;top:8px;right:10px;
  width:18px;height:18px;display:flex;align-items:center;justify-content:center;
  background:var(--gold-glow-soft);border-radius:50%;
  font-size:.68rem;font-weight:700;color:var(--gold);
}
.source-title{
  font-size:.82rem;font-weight:600;color:var(--text-primary);
  line-height:1.35;margin-bottom:6px;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
  padding-right:24px;
}
.source-meta{
  display:flex;align-items:center;gap:6px;
  font-size:.72rem;color:var(--text-muted);
}
.source-favicon{
  width:14px;height:14px;border-radius:3px;
  background:linear-gradient(135deg,var(--gold) 0%,var(--gold-deep) 100%);
  display:flex;align-items:center;justify-content:center;
  font-size:.6rem;font-weight:700;color:#1a1208;flex-shrink:0;
}

.answer-label{
  display:flex;align-items:center;gap:8px;
  font-size:.78rem;font-weight:600;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:1.5px;
  margin-bottom:14px;
}
.answer-label svg{width:14px;height:14px;color:var(--gold)}
.answer-content{
  font-size:1rem;line-height:1.75;color:var(--text-primary);
  word-wrap:break-word;
}
.answer-content p{margin-bottom:1em}
.answer-content p:last-child{margin-bottom:0}
.answer-content strong{color:var(--gold-bright);font-weight:600}
.answer-content em{color:var(--text-secondary);font-style:italic}
.answer-content code{
  font-family:'JetBrains Mono',monospace;font-size:.88em;
  padding:2px 6px;background:var(--bg-secondary);
  border:1px solid var(--border);border-radius:5px;color:var(--gold-soft);
}
.answer-content pre{
  font-family:'JetBrains Mono',monospace;font-size:.88em;
  padding:16px;background:var(--bg-secondary);
  border:1px solid var(--border);border-radius:10px;
  overflow-x:auto;margin:1em 0;
}
.answer-content ul,.answer-content ol{margin:.5em 0 1em 1.5em}
.answer-content li{margin-bottom:.4em}
.answer-content h1,.answer-content h2,.answer-content h3{
  font-family:'Cormorant Garamond',serif;
  color:var(--gold-bright);margin:1.2em 0 .5em;
  font-weight:600;
}
.answer-content h2{font-size:1.4rem}
.answer-content h3{font-size:1.15rem}

.cite{
  display:inline-flex;align-items:center;justify-content:center;
  min-width:18px;height:18px;padding:0 5px;margin:0 2px;
  background:var(--gold-glow-soft);
  border:1px solid var(--border-strong);
  border-radius:5px;
  font-size:.7rem;font-weight:700;color:var(--gold);
  cursor:pointer;transition:all .15s;
  vertical-align:1px;
  font-family:'Inter',sans-serif;
}
.cite:hover{background:var(--gold);color:#1a1208;transform:translateY(-1px)}

.rewrite-badge{
  display:inline-flex;align-items:center;gap:6px;
  padding:5px 10px;margin-bottom:14px;
  background:var(--gold-glow-soft);
  border:1px solid var(--border-strong);border-radius:999px;
  font-size:.74rem;color:var(--gold);font-weight:500;
}

/* === Pistes complémentaires (recherche approfondie) === */
.deep-block{
  margin-top:20px;
  border:1px solid var(--border);border-radius:12px;
  background:rgba(17,26,54,0.35);overflow:hidden;
}
.deep-block summary{
  display:flex;align-items:center;gap:8px;
  padding:11px 16px;cursor:pointer;list-style:none;
  font-size:.78rem;font-weight:600;color:var(--gold);
  text-transform:uppercase;letter-spacing:1.5px;
  transition:background .2s;user-select:none;
}
.deep-block summary::-webkit-details-marker{display:none}
.deep-block summary:hover{background:var(--gold-glow-soft)}
.deep-block summary svg{width:14px;height:14px;transition:transform .2s}
.deep-block[open] summary svg.chevron{transform:rotate(90deg)}
.deep-count{
  font-weight:500;text-transform:none;letter-spacing:0;
  color:var(--text-muted);font-size:.76rem;
}
.deep-status{
  display:flex;align-items:center;gap:10px;
  padding:12px 16px;border-top:1px solid var(--border);
  font-size:.85rem;color:var(--text-secondary);
}
.deep-body{padding:14px 16px 16px;border-top:1px solid var(--border)}
.deep-note{font-size:.78rem;color:var(--text-muted);margin-bottom:12px}
.deep-docs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.deep-doc-chip{
  padding:4px 10px;background:var(--gold-glow-soft);
  border:1px solid var(--border-strong);border-radius:999px;
  font-size:.74rem;color:var(--gold);word-break:break-word;
}
.cite-static{cursor:default;opacity:.65}
.cite-static:hover{background:var(--gold-glow-soft);color:var(--gold);transform:none}

.meta-info{
  display:flex;align-items:center;gap:14px;
  margin-top:10px;font-size:.72rem;color:var(--text-muted);
}

.msg-toolbar{
  display:flex;align-items:center;gap:4px;
  margin-top:18px;padding-top:14px;
  border-top:1px solid var(--border);
  flex-wrap:wrap;
}
.tool-btn{
  display:flex;align-items:center;gap:6px;
  padding:6px 10px;
  background:transparent;border:none;border-radius:7px;
  color:var(--text-muted);font-size:.78rem;font-weight:500;
  cursor:pointer;transition:all .15s;
}
.tool-btn:hover:not(:disabled){background:var(--bg-secondary);color:var(--gold)}
.tool-btn:disabled{opacity:.5;cursor:not-allowed}
.tool-btn.selected{color:var(--gold);background:var(--gold-glow-soft)}
.tool-btn svg{width:13px;height:13px}
.tool-spacer{flex:1}
.tool-rate{display:flex;gap:2px}
.feedback-status-msg{font-size:.74rem;color:var(--gold);margin-left:8px}

/* Indicateur de réflexion */
.thinking{
  display:flex;align-items:center;gap:10px;
  padding:14px 0;
  color:var(--text-secondary);font-size:.92rem;
  animation:fadeInUp .3s ease-out;
}
.thinking-dots{display:flex;gap:4px}
.thinking-dots span{
  width:7px;height:7px;border-radius:50%;
  background:var(--gold);
  box-shadow:0 0 8px var(--gold-glow);
  animation:pulse 1.4s ease-in-out infinite;
}
.thinking-dots span:nth-child(2){animation-delay:.2s}
.thinking-dots span:nth-child(3){animation-delay:.4s}

/* === COMPOSER === */
.composer-wrap{
  position:fixed;bottom:0;left:0;right:0;
  padding:16px 24px 24px;
  background:linear-gradient(180deg,transparent 0%,var(--bg-deep) 40%);
  z-index:20;
  pointer-events:none;
  transition:right .3s cubic-bezier(.4,0,.2,1);
}
.composer-wrap.with-sidebar{right:380px}
.composer-wrap > *{pointer-events:auto}
.composer{
  max-width:780px;margin:0 auto;
  background:rgba(17,26,54,0.85);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  padding:10px 10px 10px 18px;
  display:flex;align-items:flex-end;gap:10px;
  transition:all .25s;
  box-shadow:var(--shadow-card);
}
.composer:focus-within{
  border-color:var(--border-strong);
  box-shadow:0 0 0 3px var(--gold-glow-soft),var(--shadow-card);
}
.composer-input{
  flex:1;background:transparent;border:none;outline:none;
  color:var(--text-primary);font-family:inherit;font-size:.98rem;
  line-height:1.55;resize:none;
  padding:10px 0;max-height:200px;min-height:24px;
}
.composer-input::placeholder{color:var(--text-muted)}
.composer-actions{display:flex;align-items:center;gap:4px;flex-shrink:0}
.btn-send{
  width:38px;height:38px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,var(--gold) 0%,var(--gold-soft) 100%);
  border:none;border-radius:50%;
  color:#1a1208;cursor:pointer;transition:all .2s;
  box-shadow:0 0 0 0 var(--gold-glow);
}
.btn-send:hover:not(:disabled){transform:scale(1.05);box-shadow:0 0 16px var(--gold-glow)}
.btn-send:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.btn-send svg{width:16px;height:16px}

.composer-hint{
  text-align:center;font-size:.72rem;color:var(--text-muted);
  margin-top:8px;
}
.composer-hint kbd{
  font-family:'JetBrains Mono',monospace;font-size:.7rem;
  padding:1px 5px;background:var(--bg-secondary);
  border:1px solid var(--border);border-radius:4px;
  color:var(--text-secondary);
}

/* === Drawer paramètres === */
.drawer-overlay{
  position:fixed;inset:0;background:rgba(6,10,24,.6);
  backdrop-filter:blur(4px);
  opacity:0;pointer-events:none;transition:opacity .25s;
  z-index:50;
}
.drawer-overlay.open{opacity:1;pointer-events:auto}
.drawer{
  position:fixed;top:0;right:0;height:100%;width:400px;max-width:90vw;
  background:var(--bg-primary);
  border-left:1px solid var(--border);
  transform:translateX(100%);transition:transform .3s cubic-bezier(.4,0,.2,1);
  z-index:60;display:flex;flex-direction:column;
  box-shadow:-12px 0 40px rgba(0,0,0,.5);
}
.drawer.open{transform:translateX(0)}
.drawer-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 24px;border-bottom:1px solid var(--border);
}
.drawer-title{
  font-family:'Cormorant Garamond',serif;font-style:italic;font-weight:600;
  font-size:1.4rem;
  background:linear-gradient(135deg,var(--gold-bright),var(--gold));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.drawer-body{flex:1;overflow-y:auto;padding:20px 24px}
.drawer-footer{
  padding:14px 24px;border-top:1px solid var(--border);
  display:flex;gap:10px;
}
.drawer-section{margin-bottom:28px}
.drawer-section-title{
  font-size:.72rem;font-weight:700;color:var(--gold);
  text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;
}
.field{margin-bottom:16px}
.field-label{display:block;font-size:.85rem;font-weight:500;color:var(--text-secondary);margin-bottom:6px}
.field-input,.field-select,.field-textarea{
  width:100%;padding:9px 12px;
  background:var(--bg-secondary);
  border:1px solid var(--border);border-radius:9px;
  color:var(--text-primary);font-family:inherit;font-size:.9rem;
  transition:all .2s;
}
.field-input:focus,.field-select:focus,.field-textarea:focus{
  outline:none;border-color:var(--border-strong);
  box-shadow:0 0 0 3px var(--gold-glow-soft);
}
.field-textarea{resize:vertical;min-height:100px;font-family:'JetBrains Mono',monospace;font-size:.82rem}
.field-textarea:disabled{opacity:.5;cursor:not-allowed}

.toggle-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 0;
}
.toggle-info{flex:1}
.toggle-name{font-size:.9rem;font-weight:500;color:var(--text-primary)}
.toggle-desc{font-size:.78rem;color:var(--text-muted);margin-top:2px}
.toggle{
  position:relative;width:40px;height:22px;
  background:var(--bg-tertiary);border-radius:11px;
  cursor:pointer;transition:all .25s;flex-shrink:0;
  border:1px solid var(--border);
}
.toggle::after{
  content:'';position:absolute;top:2px;left:2px;
  width:16px;height:16px;border-radius:50%;
  background:var(--text-secondary);
  transition:all .25s;
}
.toggle.on{background:linear-gradient(135deg,var(--gold) 0%,var(--gold-soft) 100%);border-color:var(--gold)}
.toggle.on::after{left:20px;background:#1a1208}

.btn-primary{
  flex:1;padding:10px 14px;
  background:linear-gradient(135deg,var(--gold) 0%,var(--gold-soft) 100%);
  color:#1a1208;border:none;border-radius:10px;
  font-size:.88rem;font-weight:600;cursor:pointer;
  transition:all .2s;
}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 16px var(--gold-glow)}
.btn-secondary{
  flex:1;padding:10px 14px;
  background:transparent;color:var(--text-secondary);
  border:1px solid var(--border);border-radius:10px;
  font-size:.88rem;font-weight:500;cursor:pointer;
  transition:all .2s;
}
.btn-secondary:hover{border-color:var(--border-strong);color:var(--gold)}

/* === Modale (feedback / username) === */
.modal-overlay{
  position:fixed;inset:0;background:rgba(6,10,24,.7);
  backdrop-filter:blur(6px);
  opacity:0;pointer-events:none;transition:opacity .25s;
  z-index:70;display:flex;align-items:center;justify-content:center;
  padding:24px;
}
.modal-overlay.open,.modal-overlay.active{opacity:1;pointer-events:auto}
.modal{
  background:var(--bg-primary);border:1px solid var(--border-strong);
  border-radius:var(--radius-lg);
  width:100%;max-width:600px;max-height:85vh;
  display:flex;flex-direction:column;
  transform:scale(.96);transition:transform .25s;
  box-shadow:var(--shadow-card);
}
.modal-overlay.open .modal,.modal-overlay.active .modal{transform:scale(1)}
.modal-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 24px;border-bottom:1px solid var(--border);
}
.modal-title{
  font-family:'Cormorant Garamond',serif;font-style:italic;font-weight:600;font-size:1.3rem;
  background:linear-gradient(135deg,var(--gold-bright),var(--gold));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.modal-body{flex:1;overflow-y:auto;padding:20px 24px}
.modal-footer{
  padding:14px 24px;border-top:1px solid var(--border);
  display:flex;gap:10px;justify-content:flex-end;
}
.modal-quote{
  padding:10px 14px;background:var(--bg-secondary);
  border-left:3px solid var(--gold);border-radius:0 8px 8px 0;
  font-size:.85rem;color:var(--text-secondary);margin-bottom:14px;
  word-break:break-word;
}

/* === Toast === */
.toast{
  position:fixed;bottom:120px;left:50%;transform:translateX(-50%) translateY(20px);
  background:var(--bg-secondary);border:1px solid var(--border-strong);
  padding:10px 18px;border-radius:999px;
  font-size:.85rem;color:var(--text-primary);
  box-shadow:var(--shadow-card);
  opacity:0;pointer-events:none;transition:all .25s;
  z-index:100;
}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* === Responsive === */
@media (max-width:768px){
  .header{padding:12px 16px}
  .header-center{display:none}
  .brand-tag{display:none}
  .brand-name{font-size:1.25rem}
  .conversation{padding:24px 16px 200px}
  .user-question{font-size:1.4rem}
  .welcome{padding:24px 16px 100px}
  .welcome-logo{width:72px;height:72px}
  .composer-wrap{padding:12px 14px 16px}
  .sources-grid{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
  .composer-hint{display:none}
  .sources-sidebar{width:100%;position:fixed;inset:0;z-index:80}
  .composer-wrap.with-sidebar{right:0}
}
</style>
</head>
<body>

<div class="app">
  <!-- Bouton réouverture sidebar -->
  <button class="sb-toggle" id="sbToggle" onclick="openSidebar()" title="Sources & passages">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
    <span class="badge" id="sbBadge">0</span>
  </button>

  <!-- HEADER -->
  <header class="header">
    <div class="brand">
      <div class="brand-logo-wrap">
        <img src="/logo.png" alt="" class="brand-logo">
      </div>
      <div>
        <div class="brand-name">Luciole</div>
        <div class="brand-tag">{{SERVICE_NAME}}</div>
      </div>
    </div>

    <div class="header-center">
      <select id="index" onchange="onIndexChange()" style="display:none"></select>
    </div>

    <div class="header-actions">
      <button class="user-badge" id="userBadge" onclick="promptUserName()" style="display:none">👤</button>
      <button class="btn-new" onclick="startNewChat()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        Nouveau chat
      </button>
      <button class="icon-btn" onclick="openSettings()" title="Paramètres">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
      </button>
    </div>
  </header>

  <div class="body-row">
    <!-- MAIN -->
    <main class="main" id="main">
      <!-- ÉCRAN D'ACCUEIL -->
      <div class="welcome" id="welcome">
        <div class="welcome-logo" id="welcomeLogo">
          <img src="/logo.png" alt="Luciole">
          <span class="abdomen">
            <span class="spark s1"></span>
            <span class="spark s2"></span>
            <span class="spark s3"></span>
            <span class="spark s4"></span>
            <span class="spark s5"></span>
          </span>
        </div>
        <div class="welcome-composer">
          <div class="composer">
            <textarea class="composer-input" id="welcomeInput" placeholder="Posez votre question…" rows="1" onkeydown="handleKey(event,'welcome')" oninput="autoResize(this)"></textarea>
            <div class="composer-actions">
              <button class="btn-send" onclick="sendFromWelcome()" id="welcomeSend">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- ZONE DE CONVERSATION -->
      <div class="conversation" id="conversation" style="display:none"></div>
    </main>

    <!-- SIDEBAR DROITE -->
    <aside class="sources-sidebar" id="sourcesSidebar">
      <div class="sb-header">
        <div class="sb-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Sources
          <span class="sb-count" id="sbCount">0</span>
        </div>
        <button class="icon-btn" onclick="closeSidebar()" title="Fermer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="sb-tabs">
        <button class="sb-tab active" data-pane="sources" onclick="switchPane('sources')">Sources</button>
        <button class="sb-tab" data-pane="passages" onclick="switchPane('passages')">Passages</button>
      </div>
      <div class="sb-body">
        <div class="sb-pane active" id="paneSources">
          <div class="sb-empty">Aucune source pour le moment.</div>
        </div>
        <div class="sb-pane" id="panePassages">
          <div class="sb-empty">Aucun passage pour le moment.</div>
        </div>
      </div>
    </aside>
  </div>

  <!-- COMPOSER en bas (mode conversation) -->
  <div class="composer-wrap" id="composerWrap" style="display:none">
    <div style="position:relative;max-width:780px;margin:0 auto">
      <div class="composer">
        <textarea class="composer-input" id="mainInput" placeholder="Posez votre question…" rows="1" onkeydown="handleKey(event,'main')" oninput="autoResize(this)"></textarea>
        <div class="composer-actions">
          <button class="btn-send" onclick="sendFromMain()" id="mainSend">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
          </button>
        </div>
      </div>
      <div class="composer-hint">
        <kbd>Entrée</kbd> pour envoyer · <kbd>Maj</kbd>+<kbd>Entrée</kbd> nouvelle ligne
      </div>
    </div>
  </div>
</div>

<!-- DRAWER PARAMÈTRES -->
<div class="drawer-overlay" id="drawerOverlay" onclick="closeSettings()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <div class="drawer-title">Paramètres</div>
    <button class="icon-btn" onclick="closeSettings()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>
  <div class="drawer-body">
    <div class="drawer-section">
      <div class="drawer-section-title">Identité</div>
      <div class="field">
        <label class="field-label">Nom d'utilisateur</label>
        <input type="text" class="field-input" id="settingsUserName" placeholder="Votre prénom" onchange="updateUserNameFromSettings()">
      </div>
    </div>

    <div class="drawer-section">
      <div class="drawer-section-title">Recherche</div>
      <div class="field">
        <label class="field-label">Index actif</label>
        <select class="field-select" id="indexSelectMirror" onchange="onIndexMirrorChange()"></select>
      </div>
      <div class="field">
        <label class="field-label">Top K passages : <span id="topKLabel" style="color:var(--gold)">20</span> <span id="topKNote" style="opacity:.6;font-size:.85em">(non applicable en mode agent)</span></label>
        <select class="field-select" id="topKSelect" disabled onchange="document.getElementById('topKLabel').textContent=this.value">
          <option value="3">3</option>
          <option value="5">5</option>
          <option value="10">10</option>
          <option value="20" selected>20</option>
          <option value="30">30</option>
          <option value="50">50</option>
        </select>
      </div>
      <div class="toggle-row">
        <div class="toggle-info">
          <div class="toggle-name">Recherche approfondie</div>
          <div class="toggle-desc">Ajoute sous la réponse les pistes du profil élargi (deux passages, plus lent)</div>
        </div>
        <div class="toggle" id="enableDeepSearch" onclick="this.classList.toggle('on')"></div>
      </div>
    </div>

    <div class="drawer-section">
      <div class="drawer-section-title">Prompt système</div>
      <div class="toggle-row">
        <div class="toggle-info">
          <div class="toggle-name">Prompt personnalisé</div>
          <div class="toggle-desc">Remplace le prompt système par défaut</div>
        </div>
        <div class="toggle" id="enableCustomPrompt" onclick="this.classList.toggle('on');toggleCustomPromptUI()"></div>
      </div>
      <div class="field">
        <textarea class="field-textarea" id="customPrompt" placeholder="Prompt personnalisé (optionnel)…" disabled></textarea>
      </div>
    </div>
  </div>
  <div class="drawer-footer">
    <button class="btn-secondary" onclick="resetSettings()">Réinitialiser</button>
    <button class="btn-primary" onclick="saveSettings()">Enregistrer</button>
  </div>
</div>

<!-- MODALE FEEDBACK NEGATIF -->
<div class="modal-overlay" id="feedbackModal" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title">Aide-nous à améliorer cette réponse</div>
      <button class="icon-btn" onclick="closeModal()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <div class="modal-body">
      <div style="font-size:.78rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Question</div>
      <div class="modal-quote" id="modalQuery"></div>
      <div style="font-size:.78rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Réponse fournie</div>
      <div class="modal-quote" id="modalResponse"></div>
      <div class="field">
        <label class="field-label">Réponse attendue *</label>
        <textarea class="field-textarea" id="expectedResponse" placeholder="Décris la réponse correcte ou attendue…" style="font-family:inherit;font-size:.92rem"></textarea>
      </div>
      <div class="field">
        <label class="field-label">Commentaire (optionnel)</label>
        <textarea class="field-textarea" id="feedbackComment" placeholder="Pourquoi la réponse n'est pas satisfaisante ?" style="font-family:inherit;font-size:.92rem;min-height:60px"></textarea>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal()">Annuler</button>
      <button class="btn-primary" onclick="submitDownFeedback()">Envoyer</button>
    </div>
  </div>
</div>

<!-- MODALE PROMPT NOM UTILISATEUR -->
<div class="modal-overlay" id="namePrompt">
  <div class="modal" style="max-width:420px">
    <div class="modal-header">
      <div class="modal-title">Bienvenue sur Luciole</div>
    </div>
    <div class="modal-body">
      <p style="color:var(--text-secondary);margin-bottom:14px">Comment dois-je t'appeler ?</p>
      <div class="field">
        <input type="text" class="field-input" id="nameInput" placeholder="Ton prénom" onkeydown="if(event.key==='Enter')saveUserName()">
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-primary" onclick="saveUserName()">Continuer</button>
    </div>
  </div>
</div>

<!-- TOAST -->
<div class="toast" id="toast">Copié</div>

<script>
const AGENT_URL='/api';
let isLoading=false;
let isInConversation=false;
let isKeyUser=false;
let currentUserName='';
let feedbackKeyUsers=[];
let messageCounter=0;
const feedbackDataStore={};
let currentFeedbackData=null;

// ========== HISTORIQUE ==========
let conversationHistory=[];
const MAX_HISTORY=10;

function addToHistory(role,content){
  conversationHistory.push({role,content});
  if(conversationHistory.length>MAX_HISTORY){
    conversationHistory=conversationHistory.slice(-MAX_HISTORY);
  }
}

// ========== UTILS ==========
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

function autoResize(el){
  el.style.height='auto';
  el.style.height=Math.min(el.scrollHeight,200)+'px';
}

function handleKey(e,which){
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();
    if(which==='welcome') sendFromWelcome(); else sendFromMain();
  }
}

function scrollToBottom(){
  const m=document.getElementById('main');
  setTimeout(()=>m.scrollTo({top:m.scrollHeight,behavior:'smooth'}),50);
}

function showToast(msg){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),1800);
}

// Convertit un texte de réponse en HTML : transforme [n] en <span class="cite">n</span> et garde les sauts de ligne
function renderAnswerContent(text,cliquables=true){
  if(!text) return '';
  let html=escapeHtml(text);
  // citations [n] ou [n,m] -> spans cliquables (renvoi vers la sidebar),
  // ou statiques quand la numerotation ne correspond pas aux sources
  // affichees : la reponse deep a ses propres passages, absents de la
  // sidebar qui reste celle de la reponse principale.
  html=html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g,(m,nums)=>{
    return nums.split(/\s*,\s*/).map(n=>cliquables
      ?`<span class="cite" onclick="highlightSource(${n})">${n}</span>`
      :`<span class="cite cite-static">${n}</span>`).join('');
  });
  // paragraphes simples sur double retour à la ligne
  const parts=html.split(/\n\n+/);
  return parts.map(p=>'<p>'+p.replace(/\n/g,'<br>')+'</p>').join('');
}

// ========== STATUS ==========
function updateStatus(connected){
  const dot=document.getElementById('statusDot');
  if(!dot) return;
  if(connected) dot.classList.remove('error'); else dot.classList.add('error');
}

// ========== INDEX ==========
async function loadIndexes(){
  try{
    const response=await fetch(`${AGENT_URL}/indexes`);
    const data=await response.json();
    const select=document.getElementById('index');
    const mirror=document.getElementById('indexSelectMirror');
    select.innerHTML='';
    mirror.innerHTML='';

    // Mode mono-instance : 1 instance = 1 metier = 1 index force par INSTANCE_NAME
    if(data.single_index_mode && data.instance_name){
      const name=data.instance_name;
      const opt=document.createElement('option');
      opt.value=name; opt.textContent=name;
      select.appendChild(opt);
      const opt2=opt.cloneNode(true); mirror.appendChild(opt2);
      select.value=name;
      mirror.value=name;
      document.getElementById('currentIndex').textContent=name;
      // Masquer la ligne 'Index actif' dans le drawer (selecteur inutile)
      const mirrorField=mirror.closest('.field');
      if(mirrorField) mirrorField.style.display='none';
      // Rendre la pill non-cliquable (informative uniquement)
      const pill=document.querySelector('.index-pill');
      if(pill){ pill.onclick=null; pill.style.cursor='default'; pill.title='Instance : '+name; }
      updateStatus(true);
      return;
    }

    const validIndexes=(data.indexes||[]).filter(idx=>{
      if(idx.name==='documents' && data.indexes.length>1) return false;
      return true;
    });

    if(validIndexes.length===0){
      const opt=document.createElement('option');
      opt.value=''; opt.textContent='Aucun index disponible';
      select.appendChild(opt);
      const opt2=opt.cloneNode(true); mirror.appendChild(opt2);
      document.getElementById('currentIndex').textContent='Aucun index';
    } else {
      validIndexes.forEach(idx=>{
        const opt=document.createElement('option');
        opt.value=idx.name; opt.textContent=idx.name;
        select.appendChild(opt);
        const opt2=opt.cloneNode(true); mirror.appendChild(opt2);
      });
      const chosen=(validIndexes.length===1)?validIndexes[0].name:(data.default||validIndexes[0].name);
      select.value=chosen;
      mirror.value=chosen;
      document.getElementById('currentIndex').textContent=chosen;
    }
    updateStatus(true);
  } catch(error){
    console.error('Erreur chargement index:',error);
    document.getElementById('currentIndex').textContent='Déconnecté';
    updateStatus(false);
  }
}

function onIndexChange(){
  const v=document.getElementById('index').value;
  document.getElementById('currentIndex').textContent=v||'—';
  document.getElementById('indexSelectMirror').value=v;
}

function onIndexMirrorChange(){
  const v=document.getElementById('indexSelectMirror').value;
  document.getElementById('index').value=v;
  document.getElementById('currentIndex').textContent=v||'—';
}

function openIndexPicker(){ openSettings(); }

// ========== SETTINGS ==========
function toggleCustomPromptUI(){
  const enabled=document.getElementById('enableCustomPrompt').classList.contains('on');
  const textarea=document.getElementById('customPrompt');
  textarea.disabled=!enabled;
  if(enabled) textarea.focus();
}

function saveSettings(){
  const settings={
    customPrompt:document.getElementById('customPrompt').value,
    enableCustomPrompt:document.getElementById('enableCustomPrompt').classList.contains('on'),
    topK:parseInt(document.getElementById('topKSelect').value),
    deepSearch:document.getElementById('enableDeepSearch').classList.contains('on')
  };
  localStorage.setItem('lucioleSettings',JSON.stringify(settings));
  closeSettings();
  showToast('Paramètres sauvegardés');
}

function resetSettings(){
  document.getElementById('customPrompt').value='';
  document.getElementById('enableCustomPrompt').classList.remove('on');
  document.getElementById('customPrompt').disabled=true;
  document.getElementById('topKSelect').value='20';
  document.getElementById('topKLabel').textContent='20';
  document.getElementById('enableDeepSearch').classList.remove('on');
  localStorage.removeItem('lucioleSettings');
  showToast('Paramètres réinitialisés');
}

function loadSettings(){
  const saved=localStorage.getItem('lucioleSettings');
  if(saved){
    try{
      const settings=JSON.parse(saved);
      document.getElementById('customPrompt').value=settings.customPrompt||'';
      const cp=document.getElementById('enableCustomPrompt');
      if(settings.enableCustomPrompt) cp.classList.add('on'); else cp.classList.remove('on');
      document.getElementById('customPrompt').disabled=!settings.enableCustomPrompt;
      document.getElementById('topKSelect').value=settings.topK||20;
      document.getElementById('topKLabel').textContent=settings.topK||20;
      const ds=document.getElementById('enableDeepSearch');
      if(settings.deepSearch) ds.classList.add('on'); else ds.classList.remove('on');
    } catch(e){ console.error('Erreur chargement paramètres:',e); }
  }
}

function appliquerRoute(){
  const sel=document.getElementById('topKSelect');
  const note=document.getElementById('topKNote');
  sel.disabled=true;
  if(note) note.textContent='(pipeline iteratif, top_k pilote par le pipeline)';
}
function getSettings(){
  return {
    customPrompt:document.getElementById('enableCustomPrompt').classList.contains('on')
      ? document.getElementById('customPrompt').value : null,
    topK:parseInt(document.getElementById('topKSelect').value),
    deepSearch:document.getElementById('enableDeepSearch').classList.contains('on')
  };
}

function openSettings(){
  document.getElementById('drawer').classList.add('open');
  document.getElementById('drawerOverlay').classList.add('open');
  document.getElementById('settingsUserName').value=currentUserName||'';
}
function closeSettings(){
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawerOverlay').classList.remove('open');
}

// ========== ECRAN / CONVERSATION ==========
function enterConversation(){
  isInConversation=true;
  document.getElementById('welcome').style.display='none';
  document.getElementById('conversation').style.display='block';
  document.getElementById('composerWrap').style.display='block';
  setTimeout(()=>document.getElementById('mainInput').focus(),100);
}

function startNewChat(){
  isInConversation=false;
  conversationHistory=[];
  document.getElementById('welcome').style.display='flex';
  document.getElementById('welcomeLogo').classList.remove('thinking-mode');
  document.getElementById('conversation').style.display='none';
  document.getElementById('conversation').innerHTML='';
  document.getElementById('composerWrap').style.display='none';
  document.getElementById('welcomeInput').value='';
  closeSidebar();
  document.getElementById('sbToggle').classList.remove('show');
  document.getElementById('paneSources').innerHTML='<div class="sb-empty">Aucune source pour le moment.</div>';
  document.getElementById('panePassages').innerHTML='<div class="sb-empty">Aucun passage pour le moment.</div>';
  document.getElementById('sbCount').textContent='0';
  document.getElementById('sbBadge').textContent='0';
  document.getElementById('welcomeInput').focus();
  showToast('Nouvelle conversation démarrée');
}

// ========== ENVOI ==========
function sendFromWelcome(){
  const v=document.getElementById('welcomeInput').value.trim();
  if(!v||isLoading) return;
  document.getElementById('welcomeLogo').classList.add('thinking-mode');
  setTimeout(()=>{
    enterConversation();
    document.getElementById('welcomeInput').value='';
    askQuestion(v);
  },350);
}

function sendFromMain(){
  const v=document.getElementById('mainInput').value.trim();
  if(!v||isLoading) return;
  document.getElementById('mainInput').value='';
  autoResize(document.getElementById('mainInput'));
  askQuestion(v);
}

async function askQuestion(question){
  if(isLoading) return;
  isLoading=true;
  document.getElementById('welcomeSend').disabled=true;
  document.getElementById('mainSend').disabled=true;

  const conv=document.getElementById('conversation');

  // Bloc question
  const userBlock=document.createElement('div');
  userBlock.className='user-block';
  userBlock.innerHTML=`<h2 class="user-question">${escapeHtml(question)}</h2>`;
  conv.appendChild(userBlock);

  // Indicateur réflexion
  const thinking=document.createElement('div');
  thinking.className='thinking';
  thinking.id='thinkingIndicator';
  thinking.innerHTML='<div class="thinking-dots"><span></span><span></span><span></span></div>';
  conv.appendChild(thinking);
  scrollToBottom();

  addToHistory('user',question);

  const indexName=document.getElementById('index').value;
  const settings=getSettings();

  try{
    // La reponse principale reste TOUJOURS le profil standard : le
    // toggle « Recherche approfondie » ne la remplace pas, il ajoute
    // une seconde passe elargie affichee repliee en dessous.
    const {ok,data}=await queryAgent({
      query:question,
      index_name:indexName||null,
      top_k:settings.topK,
      custom_prompt:settings.customPrompt,
      deep_search:false,
      history:conversationHistory.slice(0,-1)
    },1800000);

    const t=document.getElementById('thinkingIndicator');
    if(t) t.remove();

    if(ok && data.response){
      addToHistory('assistant',data.response);
      const sources=data.sources||[];
      const passages=data.passages||[];
      const answerBlock=buildAnswerBlock({
        query:question,
        response:data.response,
        sources:sources,
        passages:passages,
        rewriting:data.query_rewriting||null,
        processing_time_ms:data.processing_time_ms,
        index_name:data.index_name
      });
      conv.appendChild(answerBlock);
      fillSidebar(sources,passages);
      if(passages.length>0) openSidebar();
      scrollToBottom();
      // Seconde passe elargie, lancee sans etre attendue : l'utilisateur
      // lit la reponse standard pendant qu'elle se termine.
      if(settings.deepSearch) lancerPistesProfondes(question,indexName,settings,answerBlock,passages);
    } else if(data.error){
      conv.appendChild(buildErrorBlock(typeof data.error==='string'?data.error:JSON.stringify(data.error)));
    } else if(data.detail){
      conv.appendChild(buildErrorBlock(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail)));
    } else {
      conv.appendChild(buildErrorBlock('Réponse inattendue : '+JSON.stringify(data)));
    }
  } catch(error){
    const t=document.getElementById('thinkingIndicator');
    if(t) t.remove();
    conv.appendChild(buildErrorBlock('Erreur de connexion : '+(error.message||error)));
    updateStatus(false);
  }

  isLoading=false;
  document.getElementById('welcomeSend').disabled=false;
  document.getElementById('mainSend').disabled=false;
}

function buildErrorBlock(msg){
  const block=document.createElement('div');
  block.className='answer-block';
  block.innerHTML=`<div class="answer-label" style="color:var(--error)">Erreur</div><div class="answer-content"><p style="color:var(--error)">${escapeHtml(msg)}</p></div>`;
  return block;
}

// Appel unique au proxy /api/query (qui relaie vers /api/query2).
async function queryAgent(payload,timeoutMs){
  const controller=new AbortController();
  const timeoutId=setTimeout(()=>controller.abort(),timeoutMs);
  try{
    const response=await fetch(`${AGENT_URL}/query`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal:controller.signal,
      body:JSON.stringify(payload)
    });
    return {ok:response.ok,data:await response.json()};
  }finally{
    clearTimeout(timeoutId);
  }
}

// Seconde passe « recherche approfondie » : meme question, meme
// historique, profil elargi (query2.deep). Le resultat s'affiche replie
// sous la reponse standard, jamais a sa place : la fusion est
// hierarchique, la reponse principale reste celle du profil eprouve.
// Les documents que seule la passe elargie a fait remonter sont mis en
// evidence — c'est la valeur ajoutee mesurable du mode deep.
async function lancerPistesProfondes(question,indexName,settings,answerBlock,mainPassages){
  const deep=document.createElement('details');
  deep.className='deep-block';
  deep.open=true; // la progression doit etre visible pendant le calcul
  deep.innerHTML=`
    <summary>
      <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
      Pistes complémentaires
      <span class="deep-count"></span>
    </summary>
    <div class="deep-status">
      <div class="thinking-dots"><span></span><span></span><span></span></div>
      Recherche approfondie en cours — l'entonnoir élargi est analysé…
    </div>`;
  const section=answerBlock.querySelector('.answer-section');
  (section||answerBlock).appendChild(deep);
  scrollToBottom();
  try{
    // Meme contexte que la passe standard : l'historique tel qu'il etait
    // avant la question (la reponse principale, deja empilee, est exclue).
    const {ok,data}=await queryAgent({
      query:question,
      index_name:indexName||null,
      top_k:settings.topK,
      custom_prompt:settings.customPrompt,
      deep_search:true,
      history:conversationHistory.slice(0,-2)
    },1800000);
    if(!deep.isConnected) return; // conversation reinitialisee entre-temps
    if(ok && data.response && !data.error){
      const passages=data.passages||[];
      const docsDeep=grouperParDocument(passages);
      const nomsMain=new Set(grouperParDocument(mainPassages).map(d=>d.file_name));
      const nouveaux=docsDeep.filter(d=>!nomsMain.has(d.file_name));
      const dureeS=data.processing_time_ms?Math.round(data.processing_time_ms/100)/10:null;
      deep.querySelector('.deep-count').textContent=nouveaux.length>0
        ?`· ${nouveaux.length} nouveau${nouveaux.length>1?'x':''} document${nouveaux.length>1?'s':''}`:'';
      const chips=nouveaux.length>0
        ?`<div class="deep-note">Documents que seule la recherche élargie a fait remonter :</div>
          <div class="deep-docs">${nouveaux.map(d=>`<span class="deep-doc-chip">${escapeHtml(d.file_name)}</span>`).join('')}</div>`
        :'';
      deep.querySelector('.deep-status').outerHTML=`
        <div class="deep-body">
          <div class="deep-note">Réponse du profil élargi (query2.deep) — plus de passages lus, au prix d'un bruit possible. Citations non cliquables : leur numérotation renvoie aux passages de cette passe, distincts de la sidebar.</div>
          ${chips}
          <div class="answer-content">${renderAnswerContent(data.response,false)}</div>
          ${dureeS!==null?`<div class="meta-info"><span>⏱ ${dureeS}s</span><span>${passages.length} passages</span></div>`:''}
        </div>`;
    } else {
      const raison=data&&(data.error||data.detail)||'réponse inattendue';
      deep.querySelector('.deep-status').textContent=
        'La recherche approfondie a échoué ('+raison+') — la réponse principale reste valable.';
    }
  }catch(e){
    if(deep.isConnected)
      deep.querySelector('.deep-status').textContent='Recherche approfondie injoignable : '+(e.message||e);
  }
}

// Regroupe les passages par document pour alimenter le panneau Sources.
//
// Le champ `sources` de l'API est tronque a dix entrees par
// _extract_sources (src/generation/llm.py) : un document dont les
// passages arrivaient au-dela du dixieme rang disparaissait du panneau
// alors que le generateur l'avait bien lu. `passages` contient la
// totalite du contexte, on l'agrege donc ici.
//
// Renvoie une liste triee par meilleur score, chaque entree portant le
// nom du fichier, le nombre de passages retenus et les intervalles de
// pages couverts.
function grouperParDocument(passages){
  const docs=new Map();
  (passages||[]).forEach(p=>{
    const nom=p.file_name||'Document';
    let d=docs.get(nom);
    if(!d){d={file_name:nom,count:0,score:0,pages:[]};docs.set(nom,d);}
    d.count++;
    const sc=p.score||0;
    if(sc>d.score) d.score=sc;
    const a=p.page_start||p.page||null;
    if(a!==null){
      const b=p.page_end||a;
      if(!d.pages.some(x=>x[0]===a&&x[1]===b)) d.pages.push([a,b]);
    }
  });
  return Array.from(docs.values())
    .sort((x,y)=>y.score-x.score)
    .map(d=>{
      d.pages.sort((m,n)=>m[0]-n[0]);
      d.pages_label=d.pages.map(([a,b])=>a===b?'p. '+a:'p. '+a+'-'+b).join(', ');
      d.count_label=d.count+' passage'+(d.count>1?'s':'');
      return d;
    });
}

function buildAnswerBlock(d){
  messageCounter++;
  const fbId='fb-'+messageCounter;
  const block=document.createElement('div');
  block.className='answer-block';

  // Sources grid, agregee par document a partir des passages.
  // Voir grouperParDocument : le champ renvoye par l'API est tronque a
  // dix entrees et ne reflete pas l'integralite du contexte lu.
  let sourcesHtml='';
  const docs=grouperParDocument(d.passages);
  if(docs.length>0){
    const cards=docs.map((doc,i)=>{
      const num=i+1;
      const fileName=doc.file_name;
      const fav=fileName.charAt(0).toUpperCase();
      const detail=[doc.count_label,doc.pages_label].filter(Boolean).join(' · ');
      return `<a class="source-card" onclick="highlightSource(${num})">
        <div class="source-num">${num}</div>
        <div class="source-title">${escapeHtml(fileName)}</div>
        <div class="source-meta">
          <div class="source-favicon">${escapeHtml(fav)}</div>
          <span>${escapeHtml(detail)}</span>
        </div>
      </a>`;
    }).join('');
    sourcesHtml=`<div class="sources-section">
      <div class="sources-label">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        Sources · ${docs.length} document${docs.length>1?'s':''}
      </div>
      <div class="sources-grid">${cards}</div>
    </div>`;
  }

  // Rewriting badge
  let rewriteHtml='';
  if(d.rewriting){
    const rw=(d.rewriting.rewritten||'').substring(0,80);
    rewriteHtml=`<div class="rewrite-badge">🔄 Reformulé : "${escapeHtml(rw)}${d.rewriting.rewritten&&d.rewriting.rewritten.length>80?'…':''}" <span style="opacity:.7">(${escapeHtml(d.rewriting.rule||'')})</span></div>`;
  }

  // Meta
  let metaHtml='';
  if(d.processing_time_ms){
    metaHtml=`<div class="meta-info"><span>⏱ ${d.processing_time_ms}ms</span><span>📁 ${escapeHtml(d.index_name||'—')}</span></div>`;
  }

  // Feedback (key users only)
  let feedbackHtml='';
  if(isKeyUser){
    feedbackDataStore[fbId]={
      query:d.query,
      response:d.response,
      sources:docs.map(doc=>doc.file_name),
      indexName:d.index_name||'',
      processingTime:d.processing_time_ms||0
    };
    feedbackHtml=`
      <button class="tool-btn" onclick="handleFeedback('${fbId}','up')" data-fb-up title="Bonne réponse">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
      </button>
      <button class="tool-btn" onclick="handleFeedback('${fbId}','down')" data-fb-down title="À améliorer">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>
      </button>
      <span class="feedback-status-msg" data-fb-status></span>
    `;
  }

  block.id=fbId+'-block';
  block.innerHTML=`
    ${sourcesHtml}
    <div class="answer-section">
      <div class="answer-label">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
        Réponse
      </div>
      ${rewriteHtml}
      <div class="answer-content">${renderAnswerContent(d.response)}</div>
      ${metaHtml}
      <div class="msg-toolbar" id="${fbId}">
        <button class="tool-btn" onclick="copyAnswer(this)">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copier
        </button>
        <div class="tool-spacer"></div>
        <div class="tool-rate">${feedbackHtml}</div>
      </div>
    </div>
  `;
  return block;
}

function copyAnswer(btn){
  const block=btn.closest('.answer-block');
  const t=block.querySelector('.answer-content').innerText;
  navigator.clipboard.writeText(t);
  showToast('Réponse copiée');
}

// ========== SIDEBAR ==========
function fillSidebar(sources,passages){
  const ps=document.getElementById('paneSources');
  const pp=document.getElementById('panePassages');
  const docs=grouperParDocument(passages);
  document.getElementById('sbCount').textContent=docs.length;
  document.getElementById('sbBadge').textContent=docs.length;

  if(docs.length===0){
    ps.innerHTML='<div class="sb-empty">Aucune source pour cette réponse.</div>';
  } else {
    ps.innerHTML=docs.map((doc,i)=>{
      const num=i+1;
      const meta=[];
      if(doc.pages_label) meta.push(doc.pages_label);
      meta.push(doc.count_label);
      const score=doc.score?doc.score.toFixed(2):'';
      return `<a class="sb-source" id="src-${num}" onclick="switchPane('passages')">
        <div class="sb-source-num">${num}</div>
        <div class="sb-source-body">
          <div class="sb-source-title">${escapeHtml(doc.file_name)}</div>
          <div class="sb-source-meta">
            <span>${escapeHtml(meta.join(' · '))}</span>
            ${score?`<span class="sb-source-score">${score}</span>`:''}
          </div>
        </div>
      </a>`;
    }).join('');
  }

  if(passages.length===0){
    pp.innerHTML='<div class="sb-empty">Aucun passage pour cette réponse.</div>';
  } else {
    pp.innerHTML=passages.map((p,i)=>{
      const num=i+1;
      const fileName=p.file_name||'Document';
      const meta=[];
      // Un fragment peut chevaucher plusieurs pages : on affiche la
      // plage quand page_end differe de page_start.
      if(p.page_start&&p.page_end&&p.page_end!==p.page_start) meta.push('p.'+p.page_start+'-'+p.page_end);
      else if(p.page) meta.push('p.'+p.page);
      if(p.section) meta.push(p.section);
      const score=p.score?p.score.toFixed(2):'';
      const text=(p.text||'').substring(0,800);
      return `<div class="sb-passage">
        <div class="sb-passage-meta">
          <span class="sb-passage-num">${num}</span>
          <span>${escapeHtml(fileName)}${meta.length?' · '+escapeHtml(meta.join(' · ')):''}</span>
          ${score?`<span style="margin-left:auto;color:var(--gold);font-family:'JetBrains Mono',monospace">${score}</span>`:''}
        </div>
        ${escapeHtml(text)}
      </div>`;
    }).join('');
  }
}

function openSidebar(){
  document.getElementById('sourcesSidebar').classList.add('open');
  document.getElementById('composerWrap').classList.add('with-sidebar');
  document.getElementById('sbToggle').classList.remove('show');
}
function closeSidebar(){
  document.getElementById('sourcesSidebar').classList.remove('open');
  document.getElementById('composerWrap').classList.remove('with-sidebar');
  if(isInConversation && document.getElementById('sbCount').textContent!=='0'){
    document.getElementById('sbToggle').classList.add('show');
  }
}
function switchPane(name){
  document.querySelectorAll('.sb-tab').forEach(t=>t.classList.toggle('active',t.dataset.pane===name));
  document.getElementById('paneSources').classList.toggle('active',name==='sources');
  document.getElementById('panePassages').classList.toggle('active',name==='passages');
  if(!document.getElementById('sourcesSidebar').classList.contains('open')) openSidebar();
}
function highlightSource(n){
  switchPane('sources');
  const el=document.getElementById('src-'+n);
  if(el){
    el.scrollIntoView({behavior:'smooth',block:'center'});
    el.style.transition='all .3s';
    el.style.boxShadow='0 0 0 2px var(--gold)';
    setTimeout(()=>{el.style.boxShadow=''},1400);
  }
}

// ========== USER & FEEDBACK ==========
function promptUserName(){
  document.getElementById('namePrompt').classList.add('open');
  document.getElementById('nameInput').value=currentUserName;
  setTimeout(()=>document.getElementById('nameInput').focus(),100);
}

function saveUserName(){
  const name=document.getElementById('nameInput').value.trim();
  if(!name) return;
  currentUserName=name.toLowerCase();
  localStorage.setItem('luciole_username',currentUserName);
  document.getElementById('namePrompt').classList.remove('open');
  const badge=document.getElementById('userBadge');
  badge.textContent='👤 '+name;
  badge.style.display='';
  checkKeyUser();
}

function updateUserNameFromSettings(){
  const v=document.getElementById('settingsUserName').value.trim();
  if(!v) return;
  currentUserName=v.toLowerCase();
  localStorage.setItem('luciole_username',currentUserName);
  document.getElementById('userBadge').textContent='👤 '+v;
  document.getElementById('userBadge').style.display='';
  checkKeyUser();
}

function checkKeyUser(){
  isKeyUser=feedbackKeyUsers.includes(currentUserName);
}

async function loadFeedbackConfig(){
  try{
    const resp=await fetch('/api/feedback/config');
    const data=await resp.json();
    if(data.enabled){
      feedbackKeyUsers=(data.key_users||[]).map(u=>u.toLowerCase());
    }
  } catch(e){ console.log('Feedback config not available'); }

  const saved=localStorage.getItem('luciole_username');
  if(saved){
    currentUserName=saved;
    const badge=document.getElementById('userBadge');
    badge.textContent='👤 '+saved.charAt(0).toUpperCase()+saved.slice(1);
    badge.style.display='';
    checkKeyUser();
  } else {
    promptUserName();
  }
}

function handleFeedback(fbId,type){
  const data=feedbackDataStore[fbId];
  if(!data) return;
  if(type==='down'){
    currentFeedbackData={fbId,...data};
    document.getElementById('modalQuery').textContent=data.query.substring(0,300);
    document.getElementById('modalResponse').textContent=data.response.substring(0,400);
    document.getElementById('feedbackModal').classList.add('open');
    return;
  }
  sendFeedbackToServer(fbId,{
    query:data.query,response:data.response,
    sources:JSON.stringify(data.sources),index_name:data.indexName,
    feedback:'up',processing_time_ms:data.processingTime,
    user_id:currentUserName
  });
}

async function submitDownFeedback(){
  const expected=document.getElementById('expectedResponse').value.trim();
  if(!expected){ alert('Décris la réponse attendue'); return; }
  const comment=document.getElementById('feedbackComment').value.trim();
  const data=currentFeedbackData;
  await sendFeedbackToServer(data.fbId,{
    query:data.query,response:data.response,
    sources:JSON.stringify(data.sources),index_name:data.indexName,
    feedback:'down',expected_response:expected,comment:comment,
    processing_time_ms:data.processingTime,user_id:currentUserName
  });
  closeModal();
}

async function sendFeedbackToServer(fbId,payload){
  try{
    const resp=await fetch('/api/feedback',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const result=await resp.json();
    if(result.status==='success') markFeedbackDone(fbId,payload.feedback);
  } catch(e){ console.error('Feedback error:',e); }
}

function markFeedbackDone(fbId,type){
  const container=document.getElementById(fbId);
  if(!container) return;
  container.querySelectorAll('.tool-btn[data-fb-up],.tool-btn[data-fb-down]').forEach(btn=>{
    btn.disabled=true;
    if((type==='up' && btn.hasAttribute('data-fb-up'))||(type==='down' && btn.hasAttribute('data-fb-down'))){
      btn.classList.add('selected');
    }
  });
  const status=container.querySelector('[data-fb-status]');
  if(status) status.textContent=type==='up'?'✓ Merci !':'✓ Feedback enregistré';
}

function closeModal(){
  document.getElementById('feedbackModal').classList.remove('open');
  document.getElementById('expectedResponse').value='';
  document.getElementById('feedbackComment').value='';
  currentFeedbackData=null;
}

// ========== INIT ==========
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){
    closeSettings();
    closeModal();
    document.getElementById('namePrompt').classList.remove('open');
  }
  if((e.metaKey||e.ctrlKey)&&e.key==='k'){ e.preventDefault(); openSettings(); }
  if((e.metaKey||e.ctrlKey)&&e.key==='n'){ e.preventDefault(); startNewChat(); }
});

window.addEventListener('load',()=>{
  loadIndexes();
  loadSettings();
  appliquerRoute();
  loadFeedbackConfig();
  setTimeout(()=>document.getElementById('welcomeInput').focus(),200);
});
</script>
</body>
</html>
"""
    return (html.replace("{{PAGE_TITLE}}", page_title)
                .replace("{{SERVICE_NAME}}", SERVICE_NAME.lower()))


@app.get("/api/indexes")
async def get_indexes():
    """Index disponibles.

    V4 : une instance = un index (voir docker-compose.yml). Il n'y a donc
    rien a demander au coeur RAG — le nom vient de son /api/health, et le
    front bascule en « mode index unique », sans selecteur.
    """
    nom = SERVICE_NAME
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            sante = (await client.get(f"{RAG_URL}/api/health")).json()
            nom = sante.get("instance") or nom
    except Exception as e:  # noqa: BLE001
        logger.warning(f"/api/health injoignable, nom d'instance par defaut ({e})")
    return {
        "indexes": [nom],
        "default": nom,
        "index": nom,
        "instance_name": nom,
        "single_index_mode": True,
    }


@app.post("/api/query")
async def query(request: ChatRequest):
    """Proxy vers le coeur RAG (POST /api/rag/query).

    Traduit dans les deux sens entre le front, ecrit pour la V3, et le
    contrat V4 (docs/rag-ingestion-spec.md section 7) :

      requete   query -> question, deep_search -> deep. index_name et
                top_k disparaissent : une instance V4 = un index, et la
                profondeur de l'entonnoir vient de settings.yaml, pas de
                l'appelant.
      reponse   answer -> response, citations -> sources. verdict,
                confidence, esquive, trace_id et trace sont ajoutes tels
                quels : le front les ignore aujourd'hui, ils sont
                disponibles pour la suite sans nouveau tour d'API.

    La duree est mesuree ici : le contrat V4 ne la porte pas, alors que
    le front l'affiche.
    """
    debut = time.perf_counter()
    try:
        history_list = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
        timeout = 1800.0 if request.deep_search else 1200.0
        charge = {
            "question": request.query,
            "custom_prompt": request.custom_prompt,
            "deep": request.deep_search,
            "history": history_list,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            reponse = await client.post(f"{RAG_URL}/api/rag/query", json=charge)
        ecoule = int((time.perf_counter() - debut) * 1000)
        if reponse.status_code != 200:
            detail = reponse.text[:300]
            logger.error(f"coeur RAG HTTP {reponse.status_code} : {detail}")
            return {"error": f"HTTP {reponse.status_code}",
                    "response": f"Le coeur RAG a renvoye une erreur {reponse.status_code}.",
                    "detail": detail, "processing_time_ms": ecoule}
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
    except httpx.TimeoutException:
        ecoule = int((time.perf_counter() - debut) * 1000)
        logger.error(f"coeur RAG : delai depasse apres {ecoule} ms")
        return {"error": "timeout",
                "response": "Le coeur RAG n'a pas repondu dans le delai imparti.",
                "processing_time_ms": ecoule}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Erreur d'appel au coeur RAG : {e}")
        return {"error": str(e), "response": f"Erreur: {e}",
                "processing_time_ms": int((time.perf_counter() - debut) * 1000)}


@app.get("/api/feedback/config")
async def proxy_feedback_config():
    """Config du recueil d'avis, servie par le service feedback.

    FEEDBACK_URL vide = service non deploye : on renvoie enabled=false,
    le front masque alors le bouton d'avis. Un service injoignable est
    traite pareil — l'indisponibilite du feedback ne doit jamais empecher
    de poser une question.
    """
    if not FEEDBACK_URL:
        return {"enabled": False, "key_users": []}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{FEEDBACK_URL}/api/feedback/config")
        if resp.status_code != 200:
            logger.warning(f"service feedback : HTTP {resp.status_code}, avis desactives")
            return {"enabled": False, "key_users": []}
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"service feedback injoignable, avis desactives ({e})")
        return {"enabled": False, "key_users": []}


@app.post("/api/feedback")
async def proxy_feedback(request: Request):
    """Transmet un avis au service feedback."""
    if not FEEDBACK_URL:
        return {"status": "error", "message": "Service de feedback non deploye"}
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            return (await client.post(f"{FEEDBACK_URL}/api/feedback", json=body)).json()
    except Exception as e:  # noqa: BLE001
        logger.error(f"envoi d'avis en echec : {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CHAT_PORT", 8501))
    uvicorn.run(app, host="0.0.0.0", port=port)
