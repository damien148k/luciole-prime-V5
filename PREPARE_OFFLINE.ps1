#Requires -Version 5.1
<#
.SYNOPSIS
  Prepare un package d'installation hors-ligne de Luciole Prime V4, sur une
  machine CONNECTEE a internet. Le dossier produit se copie tel quel sur la
  machine cible (disque externe, cle USB) ; l'installation s'y fait ensuite
  avec INSTALL_OFFLINE.ps1, sans aucun acces reseau.

.DESCRIPTION
  Le package contient :
    - les images Docker (ollama, qdrant, opensearch) exportees en .tar ;
    - l'image applicative luciole-v4:latest, construite ici puis exportee ;
    - le cache Ollama (LLM + modele d'embedding de l'evaluation RAGAS) ;
    - le cache HuggingFace (BGE-M3 et son reranker) ;
    - le code source, les scripts et la configuration d'exemple.

  Les modeles deja presents dans .\models\ sont REUTILISES tels quels : une
  machine qui a servi a une installation en ligne prepare donc le package
  sans rien retelecharger.

.PARAMETER GpuProfile
  Profil materiel de la machine CIBLE (et non de celle-ci) : il determine le
  modele LLM embarque. 'a5000' (24 Go, fenetre 32768), '3080ti' (12 Go,
  fenetre 16384) ou 'cpu' (modele 7B). Defaut : 3080ti.

.PARAMETER OutputDir
  Dossier de sortie. Defaut : .\offline_package

.PARAMETER SkipImages
  N'exporte pas les images Docker (utile pour rafraichir seulement le code
  et les modeles d'un package deja constitue).

.EXAMPLE
  .\PREPARE_OFFLINE.ps1 -GpuProfile a5000
  .\PREPARE_OFFLINE.ps1 -GpuProfile 3080ti -OutputDir D:\luciole_usb
#>
param(
    [ValidateSet("3080ti", "a5000", "cpu")]
    [string]$GpuProfile = "3080ti",

    [string]$OutputDir = ".\offline_package",

    [switch]$SkipImages
)

$ErrorActionPreference = "Stop"
# PowerShell 7.3+ transforme toute sortie stderr d'une commande externe en
# erreur terminante : un simple avertissement de docker suffirait alors a
# interrompre la preparation. Seul $LASTEXITCODE doit faire foi.
if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$PackageDir = $PSScriptRoot
Set-Location $PackageDir

function Write-Step { param([string]$N, [string]$M)
    Write-Host ""; Write-Host "[$N] $M" -ForegroundColor Cyan; Write-Host ("-" * 64)
}
function Write-OK   { param([string]$M) Write-Host "  [OK] $M" -ForegroundColor Green }
function Write-Warn { param([string]$M) Write-Host "  [!] $M" -ForegroundColor Yellow }
function Write-Err  { param([string]$M) Write-Host "  [ERREUR] $M" -ForegroundColor Red }

function Get-TailleGo {
    param([string]$Chemin)
    if (-not (Test-Path $Chemin)) { return 0 }
    $octets = (Get-ChildItem $Chemin -Recurse -File -ErrorAction SilentlyContinue |
               Measure-Object -Property Length -Sum).Sum
    if (-not $octets) { return 0 }
    return [math]::Round(($octets / 1GB), 1)
}

# Modele LLM par profil : doit rester aligne sur Get-GpuProfile d'INSTALL.ps1,
# sinon le package embarque un modele que l'installation cible ne cherche pas.
if ($GpuProfile -eq "cpu") {
    $LlmModel = "qwen2.5:7b-instruct-q4_K_M"
} else {
    $LlmModel = "qwen2.5:14b-instruct-q4_K_M"
}
$RagasEmbed     = "nomic-embed-text"
$EmbeddingModel = "BAAI/bge-m3"
$RerankerModel  = "BAAI/bge-reranker-v2-m3"

$ImagesDeBase = @(
    @{ Nom = "ollama/ollama:latest";                Fichier = "ollama.tar" },
    @{ Nom = "qdrant/qdrant:v1.7.4";                Fichier = "qdrant.tar" },
    @{ Nom = "opensearchproject/opensearch:2.11.0"; Fichier = "opensearch.tar" }
)

Clear-Host
Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host "  LUCIOLE V4 -- Preparation du package hors-ligne" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "A executer sur une machine AVEC acces internet et Docker demarre."
Write-Host ""
Write-Host "  Profil cible : $GpuProfile"
Write-Host "  LLM embarque : $LlmModel"
Write-Host "  Sortie       : $OutputDir"
Write-Host ""
Write-Host "  Prevoir 30 Go d'espace libre (package final ~22 Go :" -ForegroundColor Yellow
Write-Host "  les archives Docker sont compressees)." -ForegroundColor Yellow
Write-Host ""

# ---------------------------------------------------------------- 0/6 Docker
Write-Step "0/6" "Verification de Docker..."
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker n'est pas installe" }
    docker ps 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker Desktop n'est pas demarre" }
    Write-OK "Docker actif : $dockerVersion"
} catch {
    Write-Err $_.Exception.Message
    Read-Host "Appuyez sur Entree pour quitter"; exit 1
}

# ---------------------------------------------------------------- 1/6 Structure
Write-Step "1/6" "Creation du package et copie du code source..."

$dossiers = @("$OutputDir", "$OutputDir\docker_images", "$OutputDir\models\huggingface",
              "$OutputDir\models\ollama", "$OutputDir\configs", "$OutputDir\evaluation")
foreach ($d in $dossiers) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# Tout ce dont INSTALL.ps1 a besoin pour creer une instance. models\ et
# docker_images\ sont traites a part (etapes 2 a 5) : ils ne se copient pas
# depuis le depot, ils se construisent.
$aCopier = @(
    "src", "configs", "evaluation", "docs", "tests",
    "docker-compose.yml", "Dockerfile", "requirements.txt",
    "setup_bge_model.py", "pytest.ini",
    "INSTALL.ps1", "MANAGE.ps1", "INSTALL_OFFLINE.ps1", "DIAGNOSTIC.ps1", "README.md"
)
foreach ($item in $aCopier) {
    $src = Join-Path $PackageDir $item
    $dst = Join-Path $OutputDir $item
    if (-not (Test-Path $src)) { Write-Warn "$item introuvable, ignore"; continue }
    if ((Get-Item $src).PSIsContainer) {
        New-Item -ItemType Directory -Path $dst -Force | Out-Null
        Copy-Item -Path "$src\*" -Destination $dst -Recurse -Force
        # Les caches Python alourdissent le package et peuvent contenir du
        # bytecode d'une autre version : on les retire apres copie.
        Get-ChildItem -Path $dst -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") } |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        Copy-Item -Path $src -Destination $dst -Force
    }
    Write-Host "  $item"
}
Write-OK "Code source copie"

# ---------------------------------------------------------------- 2/6 Images de base
if ($SkipImages) {
    Write-Step "2/6" "Images Docker de base -- ignorees (-SkipImages)"
} else {
    Write-Step "2/6" "Telechargement et export des images Docker de base..."
    foreach ($img in $ImagesDeBase) {
        $tar = Join-Path "$OutputDir\docker_images" $img.Fichier
        if (Test-Path $tar) { Write-Host "  [deja present] $($img.Fichier)"; continue }
        Write-Host "  Telechargement de $($img.Nom)..."
        docker pull $img.Nom
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Echec du telechargement de $($img.Nom)"
            Read-Host "Appuyez sur Entree pour quitter"; exit 1
        }
        Write-Host "  Export vers $($img.Fichier)..."
        docker save -o $tar $img.Nom
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Echec de l'export de $($img.Nom)"
            Read-Host "Appuyez sur Entree pour quitter"; exit 1
        }
        Write-OK "$($img.Fichier) -- $(Get-TailleGo $tar) Go"
    }
}

# ---------------------------------------------------------------- 3/6 Image applicative
if ($SkipImages) {
    Write-Step "3/6" "Image applicative -- ignoree (-SkipImages)"
} else {
    Write-Step "3/6" "Construction et export de l'image applicative..."
    $tarApp = Join-Path "$OutputDir\docker_images" "luciole-v4.tar"
    Write-Host "  Build de luciole-v4:latest (plusieurs minutes)..."
    docker build -f "$PackageDir\Dockerfile" -t luciole-v4:latest $PackageDir
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec du build de l'image applicative"
        Read-Host "Appuyez sur Entree pour quitter"; exit 1
    }
    Write-Host "  Export vers luciole-v4.tar (image volumineuse, soyez patient)..."
    docker save -o $tarApp luciole-v4:latest
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec de l'export de l'image applicative"
        Read-Host "Appuyez sur Entree pour quitter"; exit 1
    }
    Write-OK "luciole-v4.tar -- $(Get-TailleGo $tarApp) Go"
}

# ---------------------------------------------------------------- 4/6 Modeles Ollama
Write-Step "4/6" "Modeles Ollama ($LlmModel + $RagasEmbed)..."

$ollamaSortie = Join-Path $OutputDir "models\ollama"
$ollamaLocal  = Join-Path $PackageDir "models\ollama"

# Un cache local complet est reutilise tel quel : inutile de retelecharger
# 9 Go si cette machine a deja servi a une installation en ligne.
$manifestesLocaux = @()
if (Test-Path "$ollamaLocal\models\manifests") {
    $manifestesLocaux = Get-ChildItem "$ollamaLocal\models\manifests" -Recurse -File -ErrorAction SilentlyContinue
}
$modelesVoulus = @($LlmModel, $RagasEmbed)
$manqueEnLocal = $false
foreach ($m in $modelesVoulus) {
    $base = ($m -split ":")[0]
    $trouve = $manifestesLocaux | Where-Object { $_.FullName -match [regex]::Escape($base) }
    if (-not $trouve) { $manqueEnLocal = $true }
}

$dejaDansPackage = Test-Path "$ollamaSortie\models\manifests"
if ($dejaDansPackage) {
    Write-OK "Cache Ollama deja dans le package -- $(Get-TailleGo $ollamaSortie) Go (supprimez le dossier pour le refaire)"
} elseif ((-not $manqueEnLocal) -and $manifestesLocaux.Count -gt 0) {
    Write-Host "  Cache local complet : copie sans telechargement..."
    Copy-Item -Path "$ollamaLocal\*" -Destination $ollamaSortie -Recurse -Force
    Write-OK "Cache Ollama copie -- $(Get-TailleGo $ollamaSortie) Go"
} else {
    Write-Host "  Telechargement dans un conteneur Ollama temporaire..."
    $tmpOllama = "luciole-prepare-ollama"
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    docker rm -f $tmpOllama 2>&1 | Out-Null
    $ErrorActionPreference = $prevEAP

    $cheminOllama = (Resolve-Path $ollamaSortie).Path
    docker run -d --name $tmpOllama -v "${cheminOllama}:/root/.ollama" ollama/ollama:latest | Out-Null
    Start-Sleep -Seconds 10
    foreach ($m in $modelesVoulus) {
        Write-Host "  ollama pull $m ..."
        docker exec $tmpOllama ollama pull $m
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Echec du telechargement de $m"
            $ErrorActionPreference = "Continue"
            docker rm -f $tmpOllama 2>&1 | Out-Null
            $ErrorActionPreference = $prevEAP
            Read-Host "Appuyez sur Entree pour quitter"; exit 1
        }
    }
    $ErrorActionPreference = "Continue"
    docker rm -f $tmpOllama 2>&1 | Out-Null
    $ErrorActionPreference = $prevEAP
    Write-OK "Modeles Ollama telecharges -- $(Get-TailleGo $ollamaSortie) Go"
}

# ---------------------------------------------------------------- 5/6 Modeles HuggingFace
Write-Step "5/6" "Modeles HuggingFace (BGE-M3 + reranker)..."

$hfSortie = Join-Path $OutputDir "models\huggingface"
$hfLocal  = Join-Path $PackageDir "models\huggingface"
$dossierEmb = "models--" + ($EmbeddingModel -replace "/", "--")
$dossierRer = "models--" + ($RerankerModel  -replace "/", "--")

$embLocalOk = Test-Path "$hfLocal\hub\$dossierEmb"
$rerLocalOk = Test-Path "$hfLocal\hub\$dossierRer"

$hfDejaDansPackage = (Test-Path "$hfSortie\hub\$dossierEmb") -and (Test-Path "$hfSortie\hub\$dossierRer")
if ($hfDejaDansPackage) {
    Write-OK "Cache HuggingFace deja dans le package -- $(Get-TailleGo $hfSortie) Go (supprimez le dossier pour le refaire)"
} elseif ($embLocalOk -and $rerLocalOk) {
    # Le cache HuggingFace relie snapshots\ vers blobs\ par des liens
    # symboliques. Une copie Windows qui les preserve produit, sur la
    # machine cible, des liens morts ou des fichiers vides : le modele
    # ne charge plus et l'erreur est illisible (« Expecting value: line 1
    # column 1 », un config.json vide). On resout donc les liens dans un
    # conteneur Linux, comme le fait le chemin de telechargement.
    Write-Host "  Cache local complet : copie avec resolution des liens..."
    $srcAbs = (Resolve-Path $hfLocal).Path
    $dstAbs = (Resolve-Path $hfSortie).Path
    $resolution = @(
        "set -e",
        "cp -r /src/. /output/",
        "find /output -xtype l -delete",
        "cd /output",
        "for l in `$(find . -type l); do t=`$(readlink -f `"`$l`"); if [ -e `"`$t`" ]; then cp --remove-destination `"`$t`" `"`$l`"; fi; done",
        "rm -rf /output/hub/models--*/blobs /output/hub/.locks /output/xet",
        "echo liens_restants=`$(find /output -type l | wc -l)"
    ) -join "; "
    docker run --rm -v "${srcAbs}:/src:ro" -v "${dstAbs}:/output" python:3.11-slim bash -c $resolution |
        ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec de la resolution des liens du cache HuggingFace"
        Read-Host "Appuyez sur Entree pour quitter"; exit 1
    }
    Write-OK "Cache HuggingFace copie, liens resolus -- $(Get-TailleGo $hfSortie) Go"
} else {
    # HuggingFace stocke les poids en blobs relies par des liens symboliques.
    # Windows ne les suit pas : on telecharge dans un conteneur Linux et on
    # resout les liens (cp -rL) avant de rapatrier des fichiers reels.
    Write-Host "  Telechargement dans un conteneur temporaire (liens symboliques resolus)..."
    $dlHf = "luciole-hf-download"
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    docker rm -f $dlHf 2>&1 | Out-Null

    $cheminHf = (Resolve-Path $hfSortie).Path
    docker run -d --name $dlHf -v "${cheminHf}:/output" python:3.11-slim sleep 3600 | Out-Null
    docker exec $dlHf pip install --root-user-action=ignore --no-cache-dir sentence-transformers 2>&1 |
        ForEach-Object { if ($_ -match "Successfully installed") { Write-Host "  $_" } }

    # Set-Content plutot que [System.IO.File] : compatible avec le
    # Constrained Language Mode (AppLocker/WDAC). python lit l'UTF-8 avec BOM.
    $scriptPy = Join-Path $env:TEMP "luciole_v4_dl_models.py"
    $lignes = @(
        "import os",
        "os.environ['HF_HOME'] = '/tmp/hf'",
        "os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/hf'",
        "from sentence_transformers import SentenceTransformer, CrossEncoder",
        "print('[1/2] $EmbeddingModel')",
        "m = SentenceTransformer('$EmbeddingModel', cache_folder='/tmp/hf')",
        "print('  dimension :', len(m.encode(['test'])[0]))",
        "print('[2/2] $RerankerModel')",
        "c = CrossEncoder('$RerankerModel')",
        "print('  score de test :', c.predict([('q', 'd')]))"
    )
    Set-Content -Path $scriptPy -Value $lignes -Encoding UTF8
    docker cp $scriptPy "${dlHf}:/tmp/dl.py" 2>&1 | Out-Null
    Remove-Item $scriptPy -ErrorAction SilentlyContinue

    docker exec $dlHf python /tmp/dl.py 2>&1 | ForEach-Object { Write-Host "  $_" }
    $resolution = "cp -rL /tmp/hf /tmp/hfr && rm -rf /tmp/hfr/hub/models--*/blobs /tmp/hfr/hub/models--*/.no_exist /tmp/hfr/hub/.locks /tmp/hfr/xet && cp -r /tmp/hfr/* /output/"
    docker exec $dlHf bash -c $resolution 2>&1 | ForEach-Object { Write-Host "  $_" }
    docker rm -f $dlHf 2>&1 | Out-Null
    $ErrorActionPreference = $prevEAP

    if ((-not (Test-Path "$hfSortie\hub\$dossierEmb")) -or (-not (Test-Path "$hfSortie\hub\$dossierRer"))) {
        Write-Err "Modeles HuggingFace incomplets dans $hfSortie"
        Write-Host "    Sans eux, aucune indexation ni recherche n'est possible." -ForegroundColor Red
        Write-Host "    Verifiez la connectivite du conteneur (proxy, TLS) puis relancez." -ForegroundColor Red
        Read-Host "Appuyez sur Entree pour quitter"; exit 1
    }
    Write-OK "Modeles HuggingFace telecharges -- $(Get-TailleGo $hfSortie) Go"
}

# ---------------------------------------------------------------- 6/6 Manifeste
Write-Step "6/6" "Manifeste, inventaire et verification finale..."

# Inventaire des fichiers de code (tout sauf models\ et docker_images\,
# verifies a part par leur presence et leur taille). Une copie vers une
# machine isolee peut perdre des fichiers en silence : antivirus qui met en
# quarantaine, chemin trop long, copie interrompue. L'inventaire permet a
# INSTALL_OFFLINE.ps1 de dire EXACTEMENT ce qui manque, au lieu de signaler
# un dossier entier sans expliquer pourquoi.
$inventaire = @()
$racineSortie = (Resolve-Path $OutputDir).Path
Get-ChildItem -Path $OutputDir -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $rel = $_.FullName.Substring($racineSortie.Length).TrimStart("\")
        ($rel -notlike "models\*") -and ($rel -notlike "docker_images\*") -and
        ($rel -ne "INVENTAIRE.json") -and ($rel -ne "MANIFEST.json")
    } |
    ForEach-Object {
        $rel = $_.FullName.Substring($racineSortie.Length).TrimStart("\")
        $inventaire += [ordered]@{ chemin = $rel; octets = $_.Length }
    }
$inventaire | ConvertTo-Json -Depth 3 |
    Set-Content -Path (Join-Path $OutputDir "INVENTAIRE.json") -Encoding UTF8
Write-OK "INVENTAIRE.json ecrit ($($inventaire.Count) fichiers de code)"

$nomsImages = @()
foreach ($img in $ImagesDeBase) { $nomsImages += $img.Nom }
$nomsImages += "luciole-v4:latest"

$manifeste = [ordered]@{
    version         = "4.0.0"
    genere_le       = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    machine_source  = $env:COMPUTERNAME
    profil_gpu      = $GpuProfile
    llm             = $LlmModel
    embedding_ragas = $RagasEmbed
    embedding       = $EmbeddingModel
    reranker        = $RerankerModel
    images          = $nomsImages
    fichiers_code   = $inventaire.Count
    tailles_go      = [ordered]@{
        images      = Get-TailleGo "$OutputDir\docker_images"
        ollama      = Get-TailleGo "$OutputDir\models\ollama"
        huggingface = Get-TailleGo "$OutputDir\models\huggingface"
        total       = Get-TailleGo $OutputDir
    }
}
$manifeste | ConvertTo-Json -Depth 4 |
    Set-Content -Path (Join-Path $OutputDir "MANIFEST.json") -Encoding UTF8
Write-OK "MANIFEST.json ecrit"

# Garde-fou : un package sans image applicative ou sans modeles ne sert a rien.
# Mieux vaut le dire ici que sur la machine cible, ou plus rien ne peut etre
# telecharge.
$manquants = @()
if (-not $SkipImages) {
    $tarsAttendus = @("luciole-v4.tar")
    foreach ($img in $ImagesDeBase) { $tarsAttendus += $img.Fichier }
    foreach ($f in $tarsAttendus) {
        if (-not (Test-Path (Join-Path "$OutputDir\docker_images" $f))) {
            $manquants += "docker_images\$f"
        }
    }
}
# Un lien symbolique survivant rend le cache inutilisable une fois copie
# sur la machine cible : les fichiers y arrivent vides ou morts.
$liensRestants = @(Get-ChildItem -Path "$OutputDir\models\huggingface" -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.LinkType -eq "SymbolicLink" })
if ($liensRestants.Count -gt 0) {
    $manquants += "$($liensRestants.Count) lien(s) symbolique(s) non resolu(s) dans models\huggingface"
}
if (-not (Test-Path "$OutputDir\models\huggingface\hub\$dossierEmb")) { $manquants += "models\huggingface\hub\$dossierEmb" }
if (-not (Test-Path "$OutputDir\models\huggingface\hub\$dossierRer")) { $manquants += "models\huggingface\hub\$dossierRer" }
if (-not (Test-Path "$OutputDir\models\ollama\models"))               { $manquants += "models\ollama\models" }
if (-not (Test-Path "$OutputDir\INSTALL_OFFLINE.ps1"))                { $manquants += "INSTALL_OFFLINE.ps1" }
if (-not (Test-Path "$OutputDir\src\ui"))                             { $manquants += "src\ui (interfaces chat, feedback, admin)" }

Write-Host ""
if ($manquants.Count -gt 0) {
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host "  PACKAGE INCOMPLET" -ForegroundColor Red
    Write-Host "================================================================" -ForegroundColor Red
    foreach ($m in $manquants) { Write-Host "  manquant : $m" -ForegroundColor Red }
    Write-Host ""
    Read-Host "Appuyez sur Entree pour quitter"; exit 1
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "  PACKAGE PRET" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dossier : $((Resolve-Path $OutputDir).Path)"
Write-Host "  Taille  : $(Get-TailleGo $OutputDir) Go"
Write-Host ""
Write-Host "  Sur la machine cible :" -ForegroundColor Yellow
Write-Host "    1. copier ce dossier sur un disque interne (les images se"
Write-Host "       chargent beaucoup plus vite que depuis une cle USB)"
Write-Host "    2. .\INSTALL_OFFLINE.ps1"
Write-Host ""
Set-Location $PackageDir
