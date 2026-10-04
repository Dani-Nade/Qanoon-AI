[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "build", "status", "api", "logs", "stop")]
    [string]$Command = "build",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$ComposeFile = Join-Path $PSScriptRoot "infra\docker-compose.yml"

function Invoke-Compose {
    param([string[]]$Arguments)

    & docker compose --file $ComposeFile @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed with exit code $LASTEXITCODE"
    }
}

switch ($Command) {
    "setup" {
        Invoke-Compose -Arguments @("build", "api")
        Invoke-Compose -Arguments @("up", "-d", "--wait", "postgres")
    }
    "build" {
        Invoke-Compose -Arguments @("up", "-d", "--wait", "postgres")
        Invoke-Compose -Arguments @("build", "api")
        $Arguments = @(
            "run", "--rm", "api", "python", "-m", "qanoon_ai.cli", "build-system"
        ) + $RemainingArgs
        Invoke-Compose -Arguments $Arguments
    }
    "status" {
        Invoke-Compose -Arguments @("up", "-d", "--wait", "postgres")
        Invoke-Compose -Arguments @(
            "run", "--rm", "api", "python", "-m", "qanoon_ai.cli", "system-status"
        )
    }
    "api" {
        Invoke-Compose -Arguments @("up", "-d", "--wait", "postgres", "api")
    }
    "logs" {
        Invoke-Compose -Arguments @("logs", "--follow", "--tail", "200", "api")
    }
    "stop" {
        Invoke-Compose -Arguments @("down")
    }
}
