param(
    [string]$Python = 'python',
    [string]$Config = '',
    [ValidateSet('cpu', 'cuda', 'auto')][string]$Device = 'cpu',
    [string]$BindAddress = '127.0.0.1',
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = Join-Path $packageRoot 'service/config.yaml' }
$resolvedConfig = (Resolve-Path -LiteralPath $Config).Path
Push-Location -LiteralPath $packageRoot
try {
    & $Python -B -m service.app --config $resolvedConfig --device $Device --host $BindAddress --port $Port
    $runExitCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $runExitCode
