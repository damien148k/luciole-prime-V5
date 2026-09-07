#Requires -Version 5.1
# ============================================================================
# INSTALLATION LUCIOLE V4 -- Multi-instance, installation en ligne
# ============================================================================
# Chaque projet obtient sa PROPRE installation, independante des autres :
# C:\RAG\luciole-<nom>\ (conteneurs, config, donnees, volumes Qdrant/
# OpenSearch propres). Relancer ce script avec un nouveau nom cree une
# installation supplementaire, sans toucher aux precedentes -- relancer
# avec un nom DEJA installe propose de la remplacer (elle, seulement).
#
# Les modeles lourds (embeddings BGE-M3, reranker, LLM Ollama) sont un
# CACHE PARTAGE entre toutes les instances, stocke dans ce depot
# (.\models\). Ce sont des caches de contenu adresse (HuggingFace hub,
# blobs Ollama) : les partager est sans risque et evite de retelecharger
# 10+ Go a chaque nouveau projet. Les donnees Qdrant/OpenSearch, elles,
# restent toujours propres a chaque instance.
#
# Ne gere pas l'installation hors-ligne (air-gap) : ce script telecharge
# les images Docker et les modeles depuis internet. Pas de chat/admin UI
# (T3/T4 non construites) : l'interface est l'API REST + la campagne de
# test dans evaluation/.
#
# Prerequis : Docker Desktop installe et demarre.
# ============================================================================

param(
    [string]$InstanceName = "",
    # Detecte automatiquement depuis la VRAM si non precise (voir Get-GpuProfile).
    [ValidateSet("auto", "3080ti", "a5000", "cpu")]
    [string]$GpuProfile = "auto",
    [string]$BaseInstallPath = "C:\RAG",

    # Installation hors-ligne : l'image et les modeles sont deja en place
    # (voir PREPARE_OFFLINE.ps1 puis INSTALL_OFFLINE.ps1). Le script ne
    # construit rien et ne telecharge rien ; il verifie et s'arrete net si
    # une piece manque, plutot que de tenter un acces reseau qui echouera.
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$PackageDir = $PSScriptRoot

$DefaultPorts = @{
    API        = 8000
    WATCHER    = 8090
    QDRANT     = 6333
    OPENSEARCH = 9200
    OLLAMA     = 11434
    CHAT       = 8501
    FEEDBACK   = 8503
    ADMIN      = 8080
}

# ============================================================================
# FONCTIONS
# ============================================================================

function Write-Step { param([string]$Step, [string]$Message)
    Write-Host ""; Write-Host "[$Step] $Message" -ForegroundColor Cyan; Write-Host ("-" * 60)
}
function Write-OK   { param([string]$Message) Write-Host "  [OK] $Message" -ForegroundColor Green }
function Write-Warn { param([string]$Message) Write-Host "  [!] $Message" -ForegroundColor Yellow }

function Test-InstanceName {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return $false }
    return $Name -match '^[a-z0-9][a-z0-9-]*$'
}

$script:AllocatedPorts = @()

# Ports deja reserves par les AUTRES instances installees, lus dans leur
# .env. Sans cela, une instance arretee ne se voit ni dans `docker ps` ni
# dans netstat : la nouvelle installation reprend ses ports et les deux
# deviennent inutilisables ensemble (mesure le 7 septembre 2026 : une
# instance de test a repris les six ports d'une instance a l'arret).
function Get-PortsDesAutresInstances {
    param([string]$Racine, [string]$InstanceCourante)
    $ports = @()
    if (-not (Test-Path $Racine)) { return $ports }
    Get-ChildItem -Path $Racine -Directory -Filter "luciole-*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.Name -eq "luciole-$InstanceCourante") { return }
            $envFile = Join-Path $_.FullName ".env"
            if (-not (Test-Path $envFile)) { return }
            Get-Content $envFile -ErrorAction SilentlyContinue | ForEach-Object {
                if ($_ -match '^\s*[A-Z_]*PORT\s*=\s*(\d+)') { $ports += [int]$matches[1] }
            }
        }
    return $ports
}

function Get-NextAvailablePort {
    param([int]$PreferredPort)
    $usedPorts = @()
    try {
        docker ps --format "{{.Ports}}" 2>$null | ForEach-Object {
            [regex]::Matches($_, '(?:0\.0\.0\.0|:::):(\d+)') | ForEach-Object { $usedPorts += [int]$_.Groups[1].Value }
        }
    } catch {}
    try {
        netstat -an | Select-String "LISTENING" | ForEach-Object {
            if ($_.Line -match ':(\d+)\s') { $usedPorts += [int]$matches[1] }
        }
    } catch {}
    $usedPorts = ($usedPorts + $script:AllocatedPorts + $script:PortsReserves) | Select-Object -Unique
    $port = $PreferredPort
    for ($i = 0; $i -lt 100; $i++) {
        if ($usedPorts -notcontains $port) { $script:AllocatedPorts += $port; return $port }
        $port++
    }
    throw "Aucun port disponible depuis $PreferredPort"
}

# Choisit num_ctx et le modele LLM selon la VRAM detectee (nvidia-smi).
# 14B q4_K_M pese ~9 Go de poids ; le cache KV a 32768 tokens ajoute
# ~8-10 Go sous Ollama (KV en fp16) -- une carte 12 Go (3080ti) ne peut
# donc pas tenir 14B a pleine fenetre en gardant de la marge pour
# l'embedder et le reranker (charges sur la meme carte par defaut).
# Fenetre reduite plutot que modele reduit : le mode standard de query2
# (15 passages ~ 4-5k tokens) tient largement dans 16384, et garder le
# meme modele que la cible A5000 rend les campagnes comparables. Le
# mode deep_search (30 passages) est a eviter sur cette configuration.
function Get-GpuProfile {
    param([string]$Requested)
    if ($Requested -eq "3080ti") { return @{ Model = "qwen2.5:14b-instruct-q4_K_M"; NumCtx = 16384; VramNote = "3080ti (12 Go) : fenetre reduite a 16384, eviter deep_search" } }
    if ($Requested -eq "a5000")  { return @{ Model = "qwen2.5:14b-instruct-q4_K_M"; NumCtx = 32768; VramNote = "A5000 (24 Go) : fenetre pleine 32768" } }
    if ($Requested -eq "cpu")    { return @{ Model = "qwen2.5:7b-instruct-q4_K_M";  NumCtx = 8192;  VramNote = "CPU : modele 7B, fenetre reduite" } }

    $vramMb = 0
    try {
        $out = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { $vramMb = [int]($out | Select-Object -First 1) }
    } catch {}

    if ($vramMb -eq 0) {
        Write-Warn "nvidia-smi indisponible : profil CPU applique par defaut. Precisez -GpuProfile si un GPU est present."
        return Get-GpuProfile -Requested "cpu"
    }
    if ($vramMb -ge 20000) { Write-OK "GPU detecte : $vramMb Mo VRAM -> profil A5000 (fenetre pleine)"; return Get-GpuProfile -Requested "a5000" }
    if ($vramMb -ge 10000) { Write-OK "GPU detecte : $vramMb Mo VRAM -> profil 3080ti (fenetre reduite)"; return Get-GpuProfile -Requested "3080ti" }
    Write-Warn "GPU detecte avec $vramMb Mo VRAM (< 10 Go) : profil CPU applique par securite"
    return Get-GpuProfile -Requested "cpu"
}

# ============================================================================
# DEBUT
# ============================================================================

Clear-Host
Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host "  LUCIOLE V4 -- Installation (en ligne, multi-instance)" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host ""
if ($Offline) {
    Write-Host "Mode HORS-LIGNE : aucun telechargement, aucun build." -ForegroundColor Yellow
    Write-Host ""
}
Write-Host "Cree une instance dediee a un projet dans : $BaseInstallPath\luciole-<nom>\"
Write-Host "Modeles partages entre instances (cache) : $PackageDir\models\"
Write-Host ""

# ---------------------------------------------------------------- 0/8 Docker
Write-Step "0/8" "Verification de Docker..."
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker n'est pas installe" }
    Write-OK "Docker detecte : $dockerVersion"
    docker ps 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker Desktop n'est pas demarre" }
    Write-OK "Docker Desktop est actif"
} catch {
    Write-Host "  [ERREUR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""; Read-Host "Appuyez sur Entree pour quitter"; exit 1
}

# ---------------------------------------------------------------- 1/8 Instance
Write-Step "1/8" "Nom de l'instance (= nom du projet, ex: beaumont-sud, brissy)..."

if (-not $InstanceName) {
    do {
        Write-Host ""
        $InstanceName = Read-Host "  Nom du projet ?"
        $InstanceName = $InstanceName.ToLower().Trim()
        if (-not (Test-InstanceName $InstanceName)) {
            Write-Warn "Nom invalide. Lettres minuscules, chiffres, tirets. Exemple : beaumont-sud"
            $InstanceName = ""
        }
    } while ([string]::IsNullOrWhiteSpace($InstanceName))
} elseif (-not (Test-InstanceName $InstanceName)) {
    throw "Nom d'instance invalide : '$InstanceName' (lettres minuscules, chiffres, tirets)"
}
$InstancePath = Join-Path $BaseInstallPath "luciole-$InstanceName"
Write-OK "Instance : $InstanceName"
Write-Host "  Repertoire : $InstancePath" -ForegroundColor Gray

if (Test-Path $InstancePath) {
    Write-Warn "L'instance '$InstanceName' existe deja dans $InstancePath"
    $confirm = Read-Host "  La REMPLACER ? Containers, volumes Qdrant/OpenSearch et donnees data\ de CETTE instance seront supprimes (les autres instances et les modeles partages ne sont pas touches) (oui/non)"
    if ($confirm -ne "oui") { Write-Host "  Installation annulee." -ForegroundColor Yellow; exit 0 }
    if (Test-Path (Join-Path $InstancePath "docker-compose.yml")) {
        Write-Host "  Arret et suppression des containers et volumes de '$InstanceName'..."
        Push-Location $InstancePath
        $ErrorActionPreference = "Continue"
        docker compose down -v 2>&1 | Out-Null
        $ErrorActionPreference = "Stop"
        Pop-Location
    }
    Remove-Item $InstancePath -Recurse -Force
    Write-OK "Ancienne installation '$InstanceName' supprimee"
}

# ---------------------------------------------------------------- 2/8 Ports
Write-Step "2/8" "Detection des ports disponibles..."
$script:AllocatedPorts = @()
$script:PortsReserves = Get-PortsDesAutresInstances -Racine $BaseInstallPath -InstanceCourante $InstanceName
if ($script:PortsReserves.Count -gt 0) {
    Write-Host "  $($script:PortsReserves.Count) port(s) deja reserves par d'autres instances, evites" -ForegroundColor Gray
}
$Ports = @{}
foreach ($name in @("API", "WATCHER", "QDRANT", "OPENSEARCH", "OLLAMA", "CHAT", "FEEDBACK", "ADMIN")) {
    $preferred = $DefaultPorts[$name]
    $allocated = Get-NextAvailablePort -PreferredPort $preferred
    $Ports[$name] = $allocated
    if ($allocated -eq $preferred) { Write-Host "  $($name.PadRight(12)) : $allocated" -ForegroundColor Gray }
    else { Write-Host "  $($name.PadRight(12)) : $allocated (prefere $preferred occupe -- probablement une autre instance active)" -ForegroundColor Yellow }
}
Write-OK "Ports alloues"

# ---------------------------------------------------------------- 3/8 Profil GPU
Write-Step "3/8" "Profil materiel (modele LLM et fenetre de contexte)..."
$gpu = Get-GpuProfile -Requested $GpuProfile
Write-OK $gpu.VramNote
Write-Host "  Modele  : $($gpu.Model)" -ForegroundColor Gray
Write-Host "  num_ctx : $($gpu.NumCtx)" -ForegroundColor Gray
Write-Host "  Rappel : un seul GPU ne charge en pratique qu'une instance a la fois" -ForegroundColor Gray
Write-Host "  (Ollama + embedder + reranker). Arretez l'instance active avant" -ForegroundColor Gray
Write-Host "  d'en utiliser une autre : .\MANAGE.ps1 -Action stop" -ForegroundColor Gray

# ---------------------------------------------------------------- 4/8 Structure + config
Write-Step "4/8" "Creation de l'instance et generation de la configuration..."

New-Item -ItemType Directory -Force -Path `
    $InstancePath, "$InstancePath\config", "$InstancePath\config\prompts", `
    "$InstancePath\data", "$InstancePath\backups", "$InstancePath\evaluation" | Out-Null

New-Item -ItemType Directory -Force -Path "$PackageDir\models\huggingface", "$PackageDir\models\ollama" | Out-Null

$settingsExample = Get-Content -Path "$PackageDir\configs\settings.yaml.example" -Raw -Encoding UTF8
$settingsContent = $settingsExample -replace "mrae-beaumont", $InstanceName
$settingsContent = $settingsContent -replace 'model: qwen2\.5:14b-instruct-q4_K_M', "model: $($gpu.Model)"
$settingsContent = $settingsContent -replace 'num_ctx: 32768\s*#.*', "num_ctx: $($gpu.NumCtx)               # $($gpu.VramNote)"
Set-Content -Path "$InstancePath\config\settings.yaml" -Value $settingsContent -Encoding UTF8
Write-OK "config\settings.yaml genere (instance=$InstanceName)"

Copy-Item -Path "$PackageDir\configs\prompts.yaml" -Destination "$InstancePath\config\prompts.yaml" -Force
Copy-Item -Path "$PackageDir\configs\prompts\*.md" -Destination "$InstancePath\config\prompts\" -Force -ErrorAction SilentlyContinue
Write-OK "config\prompts.yaml et config\prompts\*.md copies"

# evaluation\ est monte par-dessus /app/evaluation du conteneur (voir
# docker-compose.yml) : sans cette copie, le montage d'un dossier vide
# masquerait campagne_reference.py copie dans l'image au build.
Copy-Item -Path "$PackageDir\evaluation\*.py" -Destination "$InstancePath\evaluation\" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$PackageDir\evaluation\README.md" -Destination "$InstancePath\evaluation\" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$PackageDir\evaluation\jeu_reference.exemple.jsonl" -Destination "$InstancePath\evaluation\" -Force -ErrorAction SilentlyContinue
Write-OK "evaluation\ initialise (campagne, construction de jeu, relecture, diagnostic)"

$envContent = @"
# Luciole V4 -- Instance: $InstanceName
# Genere le : $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
INSTANCE_NAME=$InstanceName
# Isole conteneurs, reseau ET volumes de cette instance des autres
# installees (voir docker-compose.yml : name: des volumes qdrant_storage/
# opensearch_data depend de INSTANCE_NAME, mais le nom de PROJET Compose,
# lui, ne depend par defaut que du DOSSIER -- d'ou cette variable).
COMPOSE_PROJECT_NAME=luciole-$InstanceName
TZ=Europe/Paris

API_PORT=$($Ports['API'])
WATCHER_PORT=$($Ports['WATCHER'])
QDRANT_PORT=$($Ports['QDRANT'])
OPENSEARCH_PORT=$($Ports['OPENSEARCH'])
OLLAMA_PORT=$($Ports['OLLAMA'])
CHAT_PORT=$($Ports['CHAT'])
FEEDBACK_PORT=$($Ports['FEEDBACK'])
ADMIN_PORT=$($Ports['ADMIN'])
OLLAMA_CONTEXT_LENGTH=$($gpu.NumCtx)

WATCHER_ENABLED=true
WATCHER_MAX_FILE_SIZE_MB=9999
"@
Set-Content -Path "$InstancePath\.env" -Value $envContent -Encoding UTF8
Write-OK ".env genere"

# Gabarit docker-compose.yml (ce depot) -> compose pret a l'emploi de
# l'instance, avec le chemin ABSOLU du depot substitue partout ou le
# build et les modeles partages doivent le retrouver.
$packageDirForward = $PackageDir -replace '\\', '/'
$composeTemplate = Get-Content -Path "$PackageDir\docker-compose.yml" -Raw -Encoding UTF8
$composeContent = $composeTemplate -replace [regex]::Escape("__PACKAGE_DIR__"), $packageDirForward
Set-Content -Path "$InstancePath\docker-compose.yml" -Value $composeContent -Encoding UTF8
Copy-Item -Path "$PackageDir\MANAGE.ps1" -Destination "$InstancePath\MANAGE.ps1" -Force
Write-OK "docker-compose.yml et MANAGE.ps1 generes (build context -> $PackageDir)"

# ---------------------------------------------------------------- 5/8 Build + services de base
Write-Step "5/8" "Construction de l'image et demarrage d'Ollama/Qdrant/OpenSearch..."
Set-Location $InstancePath

if ($Offline) {
    $img = docker images -q luciole-v4:latest 2>$null
    if (-not $img) {
        Write-Host "  [ERREUR] Image luciole-v4:latest absente." -ForegroundColor Red
        Write-Host "    En mode hors-ligne elle doit avoir ete chargee au prealable :" -ForegroundColor Red
        Write-Host "    lancez INSTALL_OFFLINE.ps1 depuis le package, pas INSTALL.ps1 -Offline." -ForegroundColor Red
        Set-Location $PackageDir
        Read-Host "Appuyez sur Entree pour quitter"; exit 1
    }
    Write-OK "Image luciole-v4:latest deja chargee (mode hors-ligne, pas de build)"
} else {
    docker compose build rag
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERREUR] Echec du build (code $LASTEXITCODE). Corriger puis relancer : le cache Docker reprend au point d'echec." -ForegroundColor Red
        Set-Location $PackageDir
        Read-Host "Appuyez sur Entree pour quitter"; exit 1
    }
    Write-OK "Image construite (luciole-v4:latest, partagee par toutes les instances)"
}

docker compose up -d ollama qdrant opensearch
Write-Host "  Attente demarrage Ollama (15 s)..."
Start-Sleep -Seconds 15

# ---------------------------------------------------------------- 6/8 Modele LLM
Write-Step "6/8" "Modele LLM ($($gpu.Model))..."
$ollamaContainer = "luciole-ollama-$InstanceName"
$ErrorActionPreference = "Continue"
$ollamaList = docker exec $ollamaContainer ollama list 2>&1
$ErrorActionPreference = "Stop"
$modelBase = $gpu.Model.Split(":")[0]
$ragasEmbed = "nomic-embed-text"

if ($Offline) {
    # Le cache Ollama vient du package (models\ollama, monte par le
    # compose). Rien a telecharger : on verifie, et on le dit clairement
    # si le modele attendu n'y est pas.
    if ($ollamaList -match [regex]::Escape($modelBase)) {
        Write-OK "Modele $($gpu.Model) present dans le cache du package"
    } else {
        Write-Host "  [ERREUR] Modele $($gpu.Model) absent du cache Ollama." -ForegroundColor Red
        Write-Host "    Le package a-t-il ete prepare avec le meme profil GPU ?" -ForegroundColor Red
        Write-Host "    Profil demande ici : $GpuProfile" -ForegroundColor Red
        Set-Location $PackageDir
        Read-Host "Appuyez sur Entree pour quitter"; exit 1
    }
    if ($ollamaList -match [regex]::Escape($ragasEmbed)) {
        Write-OK "Modele $ragasEmbed present (evaluation RAGAS)"
    } else {
        Write-Warn "Modele $ragasEmbed absent : l'onglet RAGAS de l'admin restera indisponible."
    }
} else {
    if ($ollamaList -match [regex]::Escape($modelBase)) {
        Write-OK "Modele $($gpu.Model) deja present (cache partage -- pas de nouveau telechargement)"
    } else {
        Write-Host "  Telechargement (internet requis, plusieurs Go)..." -ForegroundColor Yellow
        docker exec $ollamaContainer ollama pull $gpu.Model
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [ERREUR] Echec du pull de $($gpu.Model)." -ForegroundColor Red
            Set-Location $PackageDir
            Read-Host "Appuyez sur Entree pour quitter"; exit 1
        }
        Write-OK "Modele $($gpu.Model) telecharge"
    }

    # Modele d'embedding de l'evaluation RAGAS (onglet RAGAS de l'admin).
    # Petit (~270 Mo) et partage entre instances comme le LLM ; sans lui,
    # RAGAS ne peut pas calculer answer_relevancy.
    if ($ollamaList -match [regex]::Escape($ragasEmbed)) {
        Write-OK "Modele $ragasEmbed deja present (cache partage)"
    } else {
        Write-Host "  Telechargement de $ragasEmbed (evaluation RAGAS, ~270 Mo)..." -ForegroundColor Yellow
        docker exec $ollamaContainer ollama pull $ragasEmbed
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Echec du pull de $ragasEmbed : l'onglet RAGAS restera indisponible (le reste fonctionne)."
        } else {
            Write-OK "Modele $ragasEmbed telecharge"
        }
    }
}

# ---------------------------------------------------------------- 7/8 Embeddings + reranker
Write-Step "7/8" "Preparation des modeles BGE-M3 (embedding) et reranker..."

$bgeDeja = Get-ChildItem -Path "$PackageDir\models\huggingface\hub\models--BAAI--bge-m3\snapshots" `
    -Filter "model.safetensors" -Recurse -ErrorAction SilentlyContinue
$rerankerDeja = Get-ChildItem -Path "$PackageDir\models\huggingface\hub\models--BAAI--bge-reranker-v2-m3\snapshots" `
    -Filter "model.safetensors" -Recurse -ErrorAction SilentlyContinue

if ($bgeDeja -and $rerankerDeja) {
    Write-OK "BGE-M3 + reranker deja presents dans le cache partage -- rien a telecharger"
} elseif ($Offline) {
    # Sans ces deux modeles rien ne peut etre indexe ni recherche : mieux
    # vaut s'arreter ici que laisser l'instance demarrer et echouer a la
    # premiere requete.
    Write-Host "  [ERREUR] BGE-M3 et/ou le reranker sont absents de $PackageDir\models\huggingface" -ForegroundColor Red
    Write-Host "    Le package offline est incomplet : refaites PREPARE_OFFLINE.ps1," -ForegroundColor Red
    Write-Host "    ou copiez le dossier models\huggingface du package a cet endroit." -ForegroundColor Red
    Set-Location $PackageDir
    Read-Host "Appuyez sur Entree pour quitter"; exit 1
} else {
    Write-Host "  Peut prendre 5 a 10 minutes (telechargement + conversion, ~3.5 Go)..." -ForegroundColor Yellow
    # Le service demarre d'abord (au repos, sans requete) : docker exec dans
    # un conteneur deja en cours evite le conflit de nom que produirait
    # 'docker compose run' face au container_name fixe du service.
    docker compose up -d rag
    Start-Sleep -Seconds 5
    $ragContainer = "luciole-rag-$InstanceName"

    $ErrorActionPreference = "Continue"
    docker exec -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 $ragContainer python setup_bge_model.py
    $bgeExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"

    if ($bgeExitCode -eq 0) {
        Write-OK "BGE-M3 + reranker prets"
    } else {
        Write-Warn "Preparation BGE-M3/reranker echouee -- verifiez : docker compose logs rag. L'API tentera un telechargement au premier appel."
    }
}

# ---------------------------------------------------------------- 8/8 Demarrage complet
Write-Step "8/8" "Demarrage complet (rag, watcher, chat, feedback, admin)..."
docker compose up -d
Write-Host "  Attente stabilisation (20 s)..."
Start-Sleep -Seconds 20

$healthOk = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$($Ports['API'])/api/health" -UseBasicParsing -TimeoutSec 15
    if ($r.StatusCode -eq 200) { $healthOk = $true }
} catch {}

# ---------------------------------------------------------------- Resume
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  INSTALLATION TERMINEE : $($InstanceName.ToUpper())" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
if ($healthOk) { Write-OK "API repond sur http://localhost:$($Ports['API'])/api/health" }
else { Write-Warn "API ne repond pas encore -- verifiez : .\MANAGE.ps1 -Action health" }
Write-Host ""
Write-Host "  Repertoire : $InstancePath" -ForegroundColor White
Write-Host "  Chat     : http://localhost:$($Ports['CHAT'])" -ForegroundColor Cyan
Write-Host "  Avis     : http://localhost:$($Ports['FEEDBACK'])" -ForegroundColor Cyan
Write-Host "  Admin    : http://localhost:$($Ports['ADMIN'])" -ForegroundColor Cyan
Write-Host "  API      : http://localhost:$($Ports['API'])" -ForegroundColor Gray
Write-Host "  Watcher  : http://localhost:$($Ports['WATCHER'])" -ForegroundColor Gray
Write-Host "  Ollama   : http://localhost:$($Ports['OLLAMA'])" -ForegroundColor Gray
Write-Host ""
Write-Host "  Pour ingerer les etudes d'impact de $InstanceName :" -ForegroundColor Yellow
Write-Host "    cd $InstancePath" -ForegroundColor White
Write-Host "    Deposez les fichiers dans data\ -- suivre : .\MANAGE.ps1 -Action logs -Service watcher" -ForegroundColor White
Write-Host ""
Write-Host "  Pour lancer la campagne de test (apres ingestion) :" -ForegroundColor Yellow
Write-Host "    1. Placez jeu_$InstanceName.jsonl dans $InstancePath\evaluation\" -ForegroundColor White
Write-Host "    2. docker exec -e JEU=/app/evaluation/jeu_$InstanceName.jsonl -e LABEL=$InstanceName ``" -ForegroundColor White
Write-Host "         -e CONSIGNE=/app/config/prompts/consigne_mrae.md luciole-rag-$InstanceName ``" -ForegroundColor White
Write-Host "         python /app/evaluation/campagne_reference.py" -ForegroundColor White
Write-Host ""
Write-Host "  Gestion (depuis $InstancePath) : .\MANAGE.ps1 -Action status | logs | health | stop | restart" -ForegroundColor Gray
Write-Host "  Ne charger qu'une instance a la fois sur ce GPU : arreter celle-ci" -ForegroundColor Gray
Write-Host "  (.\MANAGE.ps1 -Action stop) avant de demarrer une autre instance." -ForegroundColor Gray
Write-Host ""

Set-Location $PackageDir
