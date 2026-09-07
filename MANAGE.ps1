#Requires -Version 5.1
<#
.SYNOPSIS
  Gestion d'une instance Luciole V4. Ce script est copie par INSTALL.ps1
  dans le dossier de CHAQUE instance (C:\RAG\luciole-<nom>\) : lance-le
  depuis ce dossier pour agir sur cette instance precise.
.PARAMETER Action
  list | status | start | stop | restart | logs | health | urls | backup | remove | ingest
  'list' seul fonctionne depuis n'importe ou (balaie toutes les instances installees).
.PARAMETER Service
  Nom de service compose pour 'logs' (rag, watcher, chat, feedback, admin, ollama, qdrant, opensearch).
.PARAMETER Force
  Pour 'remove' : ne demande pas de confirmation.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("list", "status", "start", "stop", "restart", "logs", "health", "urls", "backup", "remove", "ingest")]
    [string]$Action,
    [string]$Service,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoDir = $PSScriptRoot
Set-Location $RepoDir

function Get-DotEnvValue {
    param([string]$Key)
    $envPath = Join-Path $RepoDir ".env"
    if (-not (Test-Path -LiteralPath $envPath)) { return $null }
    Get-Content -LiteralPath $envPath | ForEach-Object {
        if ($_ -match "^\s*$Key=(.*)$") { return $matches[1].Trim() }
    } | Select-Object -First 1
}

if ($Action -eq "list") {
    Write-Host "Instances Luciole (conteneurs luciole-rag-*, toutes machines Docker confondues) :" -ForegroundColor Cyan
    $noms = docker ps -a --format "{{.Names}}`t{{.Status}}" 2>$null | ForEach-Object {
        $parts = $_ -split "`t", 2
        if ($parts[0] -match '^luciole-rag-(.+)$') {
            [PSCustomObject]@{ Instance = $matches[1]; Etat = if ($parts.Count -gt 1) { $parts[1] } else { "" } }
        }
    }
    if (-not $noms) { Write-Host "  (aucune instance detectee)" }
    else { $noms | Format-Table -AutoSize }
    Write-Host "Un seul GPU : n'en laisser qu'une active a la fois (.\MANAGE.ps1 -Action stop depuis son dossier)." -ForegroundColor Gray
    return
}

$instanceName = Get-DotEnvValue -Key "INSTANCE_NAME"
if (-not $instanceName -and $Action -ne "urls") {
    throw ".env introuvable ou INSTANCE_NAME absent -- lancez ce script depuis le dossier d'une instance (C:\RAG\luciole-<nom>\), cree par INSTALL.ps1"
}

switch ($Action) {
    "status" {
        Write-Host "=== docker compose ps ===" -ForegroundColor Cyan
        docker compose ps -a
    }
    "start"   { docker compose start }
    "stop"    { docker compose stop }
    "restart" {
        docker compose restart
        Write-Host "  Reglages retrieval/query2/llm/prompts rechargeables sans redemarrage :" -ForegroundColor Gray
        Write-Host "    Invoke-RestMethod -Method Post -Uri http://localhost:$(Get-DotEnvValue API_PORT)/api/reload-config" -ForegroundColor Gray
    }
    "logs" {
        if ($Service) { docker compose logs -f --tail 200 $Service }
        else { docker compose logs -f --tail 200 }
    }
    "health" {
        $api = Get-DotEnvValue -Key "API_PORT"; if (-not $api) { $api = "8000" }
        Write-Host "=== Conteneurs (luciole-*-$instanceName) ===" -ForegroundColor Cyan
        docker ps -a --filter "name=luciole-$instanceName" --format "table {{.Names}}`t{{.Status}}"
        Write-Host ""
        Write-Host "=== GET http://localhost:${api}/api/health ===" -ForegroundColor Cyan
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:${api}/api/health" -UseBasicParsing -TimeoutSec 15
            Write-Host "HTTP $($r.StatusCode)"; Write-Host $r.Content
        } catch {
            Write-Host "Echec : $_" -ForegroundColor Red
        }
    }
    "urls" {
        $api = Get-DotEnvValue -Key "API_PORT"; if (-not $api) { $api = "8000" }
        $watcher = Get-DotEnvValue -Key "WATCHER_PORT"; if (-not $watcher) { $watcher = "8090" }
        $ollama = Get-DotEnvValue -Key "OLLAMA_PORT"; if (-not $ollama) { $ollama = "11434" }
        Write-Host "Instance : $instanceName"
        $chat = Get-DotEnvValue -Key "CHAT_PORT"; if (-not $chat) { $chat = "8501" }
        $fb = Get-DotEnvValue -Key "FEEDBACK_PORT"; if (-not $fb) { $fb = "8503" }
        Write-Host "  Chat         : http://localhost:${chat}"
        $adm = Get-DotEnvValue -Key "ADMIN_PORT"; if (-not $adm) { $adm = "8080" }
        Write-Host "  Avis         : http://localhost:${fb}"
        Write-Host "  Admin        : http://localhost:${adm}"
        Write-Host "  API / health : http://localhost:${api}/api/health"
        Write-Host "  RAG (POST)   : http://localhost:${api}/api/rag/query"
        Write-Host "  Watcher      : http://localhost:${watcher}"
        Write-Host "  Ollama       : http://localhost:${ollama}"
    }
    "backup" {
        $ts = Get-Date -Format "yyyyMMdd-HHmmss"
        $dest = Join-Path $RepoDir "backups\backup-$ts"
        New-Item -ItemType Directory -Path $dest -Force | Out-Null
        foreach ($d in @("data", "config", "evaluation", "feedbacks")) {
            $p = Join-Path $RepoDir $d
            if (Test-Path -LiteralPath $p) { Copy-Item -Path $p -Destination (Join-Path $dest $d) -Recurse -Force }
        }
        Write-Host "Sauvegarde creee : $dest" -ForegroundColor Green
    }
    "remove" {
        if (-not $Force) {
            $c = Read-Host "Supprimer l'instance '$instanceName' (docker compose down -v, volumes Qdrant/OpenSearch effaces) ? [o/N]"
            if ($c -notmatch '^[oOyY]') { Write-Host "Annule."; return }
        }
        docker compose down -v
        Write-Host "Instance '$instanceName' arretee et volumes supprimes. Les dossiers data/ et config/ sont conserves." -ForegroundColor Yellow
    }
    "ingest" {
        $watcher = Get-DotEnvValue -Key "WATCHER_PORT"; if (-not $watcher) { $watcher = "8090" }
        Write-Host "Le watcher surveille .\data\ automatiquement -- rien a lancer manuellement." -ForegroundColor Cyan
        Write-Host "Deposez vos fichiers dans $RepoDir\data\ puis suivez :" -ForegroundColor White
        Write-Host "  .\MANAGE.ps1 -Action logs -Service watcher" -ForegroundColor White
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:${watcher}/api/watcher/status" -TimeoutSec 10 -ErrorAction Stop
            Write-Host ($r | ConvertTo-Json -Depth 4)
        } catch {
            Write-Host "  (etat watcher indisponible pour l'instant : $_)" -ForegroundColor Yellow
        }
    }
}
