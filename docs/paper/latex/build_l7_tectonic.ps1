# Compile the current manuscript with the in-project Tectonic 0.17.0 binary.
# The binary, TeX cache, and temporary files stay under backup/ and are not committed.
# A process-local drive letter keeps the cache path short enough for Tectonic's format writer.
$ErrorActionPreference = "Stop"
$latex = $PSScriptRoot
$root = (Resolve-Path (Join-Path $latex "..\..\..")).Path
$exe = Join-Path $root "backup\local_tools\tectonic-0.17.0\bin\tectonic.exe"
if (-not (Test-Path $exe)) {
    throw "Missing $exe. Place the official prebuilt Tectonic under backup/local_tools first."
}
New-Item -ItemType Directory -Force -Path `
    (Join-Path $root "backup\local_build\l7\tectonic-cache"), `
    (Join-Path $root "backup\local_build\l7\tex"), `
    (Join-Path $root "backup\local_build\l7\tmp") | Out-Null
$letter = $null
foreach ($candidate in @("W", "V", "U", "T")) {
    if (-not (Test-Path "${candidate}:\")) {
        $letter = $candidate
        break
    }
}
if (-not $letter) {
    throw "No free drive letter for the short Tectonic path."
}
subst "${letter}:" $root
try {
    $env:TECTONIC_CACHE_DIR = "${letter}:\backup\local_build\l7\tectonic-cache"
    $env:TEMP = "${letter}:\backup\local_build\l7\tmp"
    $env:TMP = $env:TEMP
    Set-Location "${letter}:\docs\paper\latex"
    & "${letter}:\backup\local_tools\tectonic-0.17.0\bin\tectonic.exe" -X compile --keep-logs --keep-intermediates --outdir "${letter}:\backup\local_build\l7\tex" main.tex
    $code = $LASTEXITCODE
}
finally {
    Set-Location $root
    subst "${letter}:" /d | Out-Null
}
exit $code
