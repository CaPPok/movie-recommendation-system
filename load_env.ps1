# Load .env into the current PowerShell session.
#
#   .\load_env.ps1
#
# Only affects the window it runs in. A new window needs it again.
#
# The repository does not read .env by itself. The data pipeline, training,
# evaluation and inference all have to run with no AWS account and no boto3
# installed, so nothing on that path may depend on an environment file. This
# script is a deliberate manual step for the commands that do call AWS.
#
# Saved UTF-8 with BOM: Windows PowerShell 5.1 reads a .ps1 as ANSI without one,
# which corrupts every non-ASCII character and can break parsing outright.

param([string]$Path = ".env")

if (-not (Test-Path $Path)) {
    Write-Error "Khong tim thay $Path. Copy .env.example thanh .env roi dien gia tri."
    exit 1
}

$loaded = 0
foreach ($line in Get-Content $Path -Encoding UTF8) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }

    $separator = $trimmed.IndexOf("=")
    if ($separator -lt 1) { continue }

    $name = $trimmed.Substring(0, $separator).Trim()
    $value = $trimmed.Substring($separator + 1).Trim()
    if ($value -eq "") { continue }

    Set-Item -Path "Env:$name" -Value $value
    $loaded++
    # Names only. One of the values is an ARN carrying the account id.
    Write-Host "  $name"
}

Write-Host ""
Write-Host "Da nap $loaded bien vao phien hien tai."
