#Requires -Version 5.1
<#
.SYNOPSIS
  Etat reel d'une instance Luciole V4 : package, modeles, image, montages.
  Lecture seule, ne modifie rien.

.DESCRIPTION
  Repond aux trois questions qui expliquent la quasi-totalite des echecs
  d'installation hors-ligne :
    1. les fichiers de modeles du package sont-ils valides, ou vides ?
    2. sur quel dossier de modeles l'instance est-elle branchee ?
    3. le conteneur tourne-t-il bien sur l'image attendue ?

.PARAMETER InstancePath
  Dossier de l'instance. Defaut : detecte sous C:\RAG.

.PARAMETER PackagePath
  Dossier du package hors-ligne. Defaut : lu dans le compose de l'instance.

.EXAMPLE
  .\DIAGNOSTIC.ps1
  .\DIAGNOSTIC.ps1 -InstancePath C:\RAG\luciole-achery
#>
param(
    [string]$InstancePath = "",
    [string]$PackagePath = ""
)

$ErrorActionPreference = "Continue"
if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Titre { param([string]$M)
    Write-Host ""; Write-Host $M -ForegroundColor Cyan; Write-Host ("-" * 64)
}
function OK   { param([string]$M) Write-Host "  [OK] $M" -ForegroundColor Green }
function KO   { param([string]$M) Write-Host "  [KO] $M" -ForegroundColor Red }
function Info { param([string]$M) Write-Host "  $M" -ForegroundColor Gray }

function Taille-Reelle {
    <#
      Taille du CONTENU d'un fichier, lien symbolique suivi.
      Get-ChildItem rapporte 0 pour un lien : le cache HuggingFace en est
      plein, et s'y fier ferait passer une installation saine pour cassee.
      L'ouverture du flux, elle, traverse le lien ; -1 signale un fichier
      illisible (lien mort, droits refuses).
    #>
    param([string]$Chemin)
    try {
        $flux = [System.IO.File]::OpenRead($Chemin)
        try { return $flux.Length } finally { $flux.Dispose() }
    } catch { return -1 }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host "  LUCIOLE V4 -- Diagnostic d'instance (lecture seule)" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta

# ---------------------------------------------------------------- Instance
Titre "1. Instance"
if (-not $InstancePath) {
    $candidats = @(Get-ChildItem "C:\RAG" -Directory -Filter "luciole-*" -ErrorAction SilentlyContinue)
    if ($candidats.Count -eq 1) {
        $InstancePath = $candidats[0].FullName
    } elseif ($candidats.Count -gt 1) {
        Info "Plusieurs instances : $($candidats.Name -join ', ')"
        Info "Relancez avec -InstancePath pour en choisir une. Premiere prise :"
        $InstancePath = $candidats[0].FullName
    }
}
if (-not $InstancePath -or -not (Test-Path $InstancePath)) {
    KO "Aucune instance trouvee. Passez -InstancePath."
    exit 1
}
$nomInstance = (Split-Path $InstancePath -Leaf) -replace "^luciole-", ""
OK "Instance '$nomInstance' : $InstancePath"

$composePath = Join-Path $InstancePath "docker-compose.yml"
if (-not (Test-Path $composePath)) { KO "docker-compose.yml absent"; exit 1 }

# ---------------------------------------------------------------- Montages
Titre "2. Ou l'instance cherche-t-elle ses modeles ?"
$compose = Get-Content $composePath -Raw -Encoding UTF8
$cheminHf = $null
$cheminOllama = $null
foreach ($ligne in ($compose -split "`n")) {
    if ($ligne -match '^\s*-\s*(.+?)/models/huggingface:/app/models/huggingface') { $cheminHf = $matches[1].Trim() }
    if ($ligne -match '^\s*-\s*(.+?)/models/ollama:/root/\.ollama')                { $cheminOllama = $matches[1].Trim() }
}
if ($cheminHf) {
    Info "modeles HuggingFace : $cheminHf/models/huggingface"
    if (Test-Path "$cheminHf/models/huggingface") { OK "ce dossier existe" }
    else { KO "CE DOSSIER N'EXISTE PAS -- le package a ete deplace ou renomme depuis l'installation" }
} else { KO "Montage des modeles HuggingFace introuvable dans le compose" }
if ($cheminOllama) { Info "modeles Ollama      : $cheminOllama/models/ollama" }

if (-not $PackagePath) { $PackagePath = $cheminHf }

# ---------------------------------------------------------------- Fichiers
Titre "3. Les fichiers de modeles sont-ils valides ?"
$problemes = 0
$dossierModeles = "$PackagePath\models\huggingface"

if (-not (Test-Path $dossierModeles)) {
    KO "Dossier de modeles introuvable : $dossierModeles"
    $problemes++
} else {
    # Controle DANS un conteneur : le cache est cree par un conteneur
    # Linux, ses liens sont des points de reparse que Windows ne suit pas.
    # Vu de l'hote, un cache sain paraitrait illisible. Le conteneur voit
    # ce que verra le watcher en production.
    $abs = (Resolve-Path $dossierModeles).Path -replace "\\", "/"
    # Pas de guillemets doubles ni de '|' dans ce script : PowerShell 5.1
    # les mange en les transmettant a docker, et sh voyait des tubes la ou
    # on voulait des separateurs (symptome : "sh: 1: 687: not found").
    # Les chemins du cache n'ont pas d'espace, les guillemets sont inutiles.
    $script = 'for m in models--BAAI--bge-m3 models--BAAI--bge-reranker-v2-m3; do ' +
              'd=$(ls -d /m/hub/$m/snapshots/*/ 2>/dev/null | head -1); ' +
              'if [ ${d:-none} = none ]; then echo $m,ABSENT,0,0,0; continue; fi; ' +
              'n=$(find -L $d -type f | wc -l); ' +
              'v=$(find -L $d -type f -size 0 | wc -l); ' +
              'c=$(stat -Lc %s ${d}config.json 2>/dev/null || echo -1); ' +
              'p=$(stat -Lc %s ${d}model.safetensors 2>/dev/null || echo -1); ' +
              'echo $m,$n,$v,$c,$p; done'
    $lignes = docker run --rm -v "${abs}:/m:ro" python:3.11-slim sh -c $script 2>$null

    if (-not $lignes) {
        KO "Controle impossible : docker n'a pas pu monter $dossierModeles"
        $problemes++
    }
    foreach ($l in $lignes) {
        if ($l -notmatch ',') { continue }
        $p = $l -split ','
        $nom, $nb, $vides, $cfg, $poids = $p[0], $p[1], $p[2], [int64]$p[3], [int64]$p[4]
        $court = $nom -replace "models--BAAI--", ""
        if ($nb -eq "ABSENT") { KO "$court : dossier snapshots absent"; $problemes++; continue }
        Info "$court : $nb fichiers"
        if ($cfg -gt 0)  { OK "config.json : $cfg octets" }
        else             { KO "config.json vide ou absent -- cause de 'Expecting value: line 1 column 1'"; $problemes++ }
        if ($poids -gt 1000000) { OK ("model.safetensors : {0:N2} Go" -f ($poids / 1GB)) }
        else                    { KO "model.safetensors vide ou absent"; $problemes++ }
        if ([int]$vides -gt 0)  { KO "$vides fichier(s) vide(s) -- copie incomplete"; $problemes++ }
    }

    # Liens symboliques : sains ici, mais ils ne survivront pas a une copie
    # vers une autre machine. Avertissement, pas defaut.
    $liens = docker run --rm -v "${abs}:/m:ro" python:3.11-slim sh -c "find /m -type l | wc -l" 2>$null
    $liens = ($liens | Where-Object { $_ -match '^\d+$' } | Select-Object -First 1)
    if ($liens -and [int]$liens -gt 0) {
        Write-Host "  [!] $liens lien(s) symbolique(s) dans ce cache." -ForegroundColor Yellow
        Write-Host "      Ils fonctionnent ici, mais une copie vers une autre machine" -ForegroundColor Yellow
        Write-Host "      les transformera en fichiers vides. Pour un package a" -ForegroundColor Yellow
        Write-Host "      transporter, utilisez PREPARE_OFFLINE.ps1, qui les resout." -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- Image
Titre "4. Image et conteneurs"
$img = docker images luciole-v4:latest --format "{{.ID}} {{.CreatedSince}} {{.Size}}" 2>$null
if ($img) { OK "image luciole-v4:latest : $img" } else { KO "image luciole-v4:latest absente" ; $problemes++ }

$conteneurs = docker ps -a --filter "name=luciole-" --format "{{.Names}}|{{.Status}}|{{.Image}}" 2>$null
if ($conteneurs) {
    foreach ($c in $conteneurs) {
        $p = $c -split '\|'
        Info ("{0,-34} {1,-22} {2}" -f $p[0], $p[1], $p[2])
    }
} else { Info "aucun conteneur luciole-*" }

# Le code reellement embarque dans l'image (permet de voir si le
# rechargement a bien eu lieu).
$sonde = docker run --rm luciole-v4:latest sh -c "grep -c 'app/backups/excel_data.db' /app/src/ingestion/sql_storage.py 2>/dev/null || echo 0" 2>$null
$sonde = ($sonde | Select-Object -Last 1).Trim()
if ($sonde -eq "0") {
    KO "L'image ne contient PAS le correctif Excel : c'est une ancienne image"
    $problemes++
} else {
    OK "L'image contient le correctif Excel"
}

# ---------------------------------------------------------------- Vu du conteneur
Titre "5. Ce que le conteneur voit reellement"
$nomWatcher = "luciole-watcher-$nomInstance"
$actif = docker ps --filter "name=$nomWatcher" --format "{{.Names}}" 2>$null
if ($actif) {
    # stat -Lc suit le lien : c'est la taille du contenu, celle qui compte.
    $vu = docker exec $nomWatcher sh -c "stat -Lc %s /app/models/huggingface/hub/models--BAAI--bge-m3/snapshots/*/config.json 2>/dev/null | head -1" 2>$null
    $vu = ($vu | Where-Object { $_ -match '^\d+$' } | Select-Object -First 1)
    if ($vu) {
        if ([int]$vu -gt 0) { OK "config.json vu par le conteneur : $vu octets" }
        else { KO "config.json vu par le conteneur : 0 octet -- le montage pointe sur des fichiers vides"; $problemes++ }
    } else { KO "config.json introuvable depuis le conteneur"; $problemes++ }
} else {
    Info "$nomWatcher n'est pas demarre : controle du montage impossible"
}

# ---------------------------------------------------------------- Verdict
Titre "Verdict"
if ($problemes -eq 0) {
    Write-Host "  Aucun probleme detecte." -ForegroundColor Green
} else {
    Write-Host "  $problemes probleme(s) detecte(s) -- voir les lignes [KO] ci-dessus." -ForegroundColor Red
}
Write-Host ""
