#Requires -Version 5.1
<#
.SYNOPSIS
  Installe Luciole Prime V4 sur une machine SANS acces internet, depuis un
  package prepare par PREPARE_OFFLINE.ps1.

.DESCRIPTION
  Deroulement :
    1. verification du package (manifeste, images, modeles) ;
    2. chargement des images Docker depuis docker_images\*.tar ;
    3. appel d'INSTALL.ps1 -Offline, qui cree l'instance sans rien
       construire ni telecharger.

  Ce script ne touche jamais au reseau. S'il manque une piece, il s'arrete
  et dit laquelle : sur une machine isolee, rien ne pourra la rattraper.

.PARAMETER InstanceName
  Nom du projet (minuscules, chiffres, tirets). Demande si absent.

.PARAMETER GpuProfile
  auto (defaut), 3080ti, a5000 ou cpu. Doit correspondre au profil utilise
  pour preparer le package, sinon le modele LLM attendu sera introuvable.

.PARAMETER BaseInstallPath
  Racine des instances. Defaut : C:\RAG

.PARAMETER Force
  Recharge les images meme si elles sont deja presentes dans Docker.

.EXAMPLE
  .\INSTALL_OFFLINE.ps1
  .\INSTALL_OFFLINE.ps1 -InstanceName brissy -GpuProfile a5000
#>
param(
    [string]$InstanceName = "",

    [ValidateSet("auto", "3080ti", "a5000", "cpu")]
    [string]$GpuProfile = "auto",

    [string]$BaseInstallPath = "C:\RAG",

    [switch]$Force
)

$ErrorActionPreference = "Stop"
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
function Quitter    { Read-Host "Appuyez sur Entree pour quitter"; exit 1 }

Clear-Host
Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host "  LUCIOLE V4 -- Installation hors-ligne" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host ""

# ---------------------------------------------------------------- 1/4 Package
Write-Step "1/4" "Verification du package..."

$manifestePath = Join-Path $PackageDir "MANIFEST.json"
$manifeste = $null
if (Test-Path $manifestePath) {
    try {
        $manifeste = Get-Content $manifestePath -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-OK "Package du $($manifeste.genere_le), profil $($manifeste.profil_gpu)"
        Write-Host "  LLM embarque : $($manifeste.llm)" -ForegroundColor Gray
    } catch {
        Write-Warn "MANIFEST.json illisible : verification poursuivie sans lui"
    }
} else {
    Write-Warn "MANIFEST.json absent : ce dossier a-t-il ete produit par PREPARE_OFFLINE.ps1 ?"
}

# Les pieces sans lesquelles l'installation ne peut pas aboutir. On les
# verifie toutes avant de commencer, pour ne pas echouer au milieu.
$requis = @(
    @{ Chemin = "INSTALL.ps1";                                  Quoi = "script d'installation" },
    @{ Chemin = "docker-compose.yml";                           Quoi = "definition des services" },
    @{ Chemin = "src\ui";                                       Quoi = "interfaces chat, feedback et admin" },
    @{ Chemin = "configs\settings.yaml.example";                Quoi = "configuration d'exemple" },
    @{ Chemin = "models\huggingface\hub\models--BAAI--bge-m3";  Quoi = "modele d'embedding BGE-M3" },
    @{ Chemin = "models\huggingface\hub\models--BAAI--bge-reranker-v2-m3"; Quoi = "reranker" },
    @{ Chemin = "models\ollama\models";                         Quoi = "cache Ollama (LLM)" }
)
$manquants = @()
foreach ($r in $requis) {
    if (-not (Test-Path (Join-Path $PackageDir $r.Chemin))) {
        $manquants += "$($r.Chemin)  ($($r.Quoi))"
    }
}

# Verification fichier par fichier contre l'inventaire produit a la
# preparation. Une copie vers une machine isolee perd parfois des fichiers
# en silence : mise en quarantaine par l'antivirus, chemin trop long, copie
# interrompue. Sans ce controle, on ne sait que « il manque un dossier ».
$absents = @()
$tronques = @()
$inventairePath = Join-Path $PackageDir "INVENTAIRE.json"
if (Test-Path $inventairePath) {
    try {
        $inventaire = Get-Content $inventairePath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($f in $inventaire) {
            $chemin = Join-Path $PackageDir $f.chemin
            if (-not (Test-Path $chemin)) {
                $absents += $f.chemin
            } elseif ((Get-Item $chemin).Length -ne $f.octets) {
                $tronques += "$($f.chemin)  (attendu $($f.octets) octets, trouve $((Get-Item $chemin).Length))"
            }
        }
        if ($absents.Count -eq 0 -and $tronques.Count -eq 0) {
            Write-OK "$($inventaire.Count) fichiers de code verifies, aucun manquant"
        }
    } catch {
        Write-Warn "INVENTAIRE.json illisible : verification detaillee ignoree"
    }
} else {
    Write-Warn "INVENTAIRE.json absent : package produit par une version anterieure du script"
}

if ($manquants.Count -gt 0 -or $absents.Count -gt 0 -or $tronques.Count -gt 0) {
    Write-Err "Package incomplet sur CETTE machine."
    if ($manquants.Count -gt 0) {
        Write-Host ""
        Write-Host "  Elements essentiels absents :" -ForegroundColor Red
        foreach ($m in $manquants) { Write-Host "    $m" -ForegroundColor Red }
    }
    if ($absents.Count -gt 0) {
        Write-Host ""
        Write-Host "  $($absents.Count) fichier(s) absent(s), dont :" -ForegroundColor Red
        foreach ($a in ($absents | Select-Object -First 15)) { Write-Host "    $a" -ForegroundColor Red }
        if ($absents.Count -gt 15) { Write-Host "    ... et $($absents.Count - 15) autres" -ForegroundColor Red }
    }
    if ($tronques.Count -gt 0) {
        Write-Host ""
        Write-Host "  $($tronques.Count) fichier(s) de taille incorrecte, dont :" -ForegroundColor Red
        foreach ($t in ($tronques | Select-Object -First 10)) { Write-Host "    $t" -ForegroundColor Red }
    }
    Write-Host ""
    Write-Host "  Le package d'origine est probablement intact : c'est la COPIE" -ForegroundColor Yellow
    Write-Host "  vers cette machine qui a perdu des fichiers. Causes frequentes :" -ForegroundColor Yellow
    Write-Host "    - antivirus : verifiez l'historique de protection de Windows" -ForegroundColor Yellow
    Write-Host "      Securite (les gros fichiers .py contiennent du HTML et du" -ForegroundColor Yellow
    Write-Host "      JavaScript, parfois mis en quarantaine)" -ForegroundColor Yellow
    Write-Host "    - chemin trop long : rapprochez le package de la racine," -ForegroundColor Yellow
    Write-Host "      par exemple C:\luciole_offline" -ForegroundColor Yellow
    Write-Host "    - cle en FAT32 : les fichiers de plus de 4 Go ne passent pas" -ForegroundColor Yellow
    Write-Host "      (luciole-v4.tar en fait environ 5) -- utilisez NTFS ou exFAT" -ForegroundColor Yellow
    Write-Host "    - copie interrompue : refaites-la avec robocopy, qui signale" -ForegroundColor Yellow
    Write-Host "      les echecs au lieu de les passer sous silence :" -ForegroundColor Yellow
    Write-Host "        robocopy <source> <destination> /E /R:2 /W:2" -ForegroundColor Gray
    Write-Host ""
    Quitter
}
Write-OK "Toutes les pieces attendues sont presentes"

# ---------------------------------------------------------------- 2/4 Docker
Write-Step "2/4" "Verification de Docker..."
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker n'est pas installe" }
    docker ps 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker Desktop n'est pas demarre" }
    Write-OK "Docker actif : $dockerVersion"
} catch {
    Write-Err $_.Exception.Message
    Quitter
}

# ---------------------------------------------------------------- 3/4 Images
Write-Step "3/4" "Chargement des images Docker..."

$imagesAttendues = @(
    @{ Fichier = "luciole-v4.tar"; Tag = "luciole-v4:latest" },
    @{ Fichier = "ollama.tar";     Tag = "ollama/ollama:latest" },
    @{ Fichier = "qdrant.tar";     Tag = "qdrant/qdrant:v1.7.4" },
    @{ Fichier = "opensearch.tar"; Tag = "opensearchproject/opensearch:2.11.0" }
)

foreach ($img in $imagesAttendues) {
    $tar = Join-Path $PackageDir "docker_images\$($img.Fichier)"
    $deja = docker images -q $img.Tag 2>$null

    if ($deja -and -not $Force) {
        Write-OK "$($img.Tag) deja chargee (-Force pour recharger)"
        continue
    }
    if (-not (Test-Path $tar)) {
        Write-Err "$($img.Fichier) absent du package et image absente de Docker."
        Write-Host "    Sans cette image, l'installation ne peut pas aboutir." -ForegroundColor Red
        Quitter
    }
    $tailleGo = "{0:N1}" -f ((Get-Item $tar).Length / 1GB)
    Write-Host "  Chargement de $($img.Fichier) ($tailleGo Go), cela peut prendre plusieurs minutes..."
    docker load -i $tar
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec du chargement de $($img.Fichier)"
        Quitter
    }
    Write-OK "$($img.Tag) chargee"
}

# ---------------------------------------------------------------- 4/4 Instance
Write-Step "4/4" "Creation de l'instance (mode hors-ligne)..."

# Le profil du package fait autorite : installer avec un autre profil
# chercherait un modele LLM que le cache ne contient pas.
if ($GpuProfile -eq "auto" -and $manifeste -and $manifeste.profil_gpu) {
    $GpuProfile = $manifeste.profil_gpu
    Write-Host "  Profil repris du package : $GpuProfile" -ForegroundColor Gray
}

$arguments = @{
    GpuProfile      = $GpuProfile
    BaseInstallPath = $BaseInstallPath
    Offline         = $true
}
if ($InstanceName) { $arguments["InstanceName"] = $InstanceName }

& (Join-Path $PackageDir "INSTALL.ps1") @arguments
