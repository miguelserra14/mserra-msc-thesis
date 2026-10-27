# Small end-to-end Joern run: source code -> CPG -> exported graphs -> queries.
# Usage (from this folder):  .\run.ps1

$JoernDir = Join-Path $HOME "joern-cli"
$Here     = $PSScriptRoot
$Out      = Join-Path $Here "out"
$Cpg      = Join-Path $Out "cpg.bin"

if (-not (Test-Path (Join-Path $JoernDir "joern-parse.bat"))) {
    Write-Error "Joern not found in $JoernDir"
    exit 1
}

# Start clean: joern-export refuses to write into an existing folder.
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory $Out | Out-Null

# Step 1 - parse: C source -> Code Property Graph (binary file).
# ("-cp requires class path specification"), so the C frontend is called directly.
Write-Host "`n[1/3] c2cpg (what joern-parse runs for C code)" -ForegroundColor Cyan
$C2cpgJar = (Get-ChildItem (Join-Path $JoernDir "frontends\c2cpg\lib\io.joern.c2cpg-*-launcher.jar")).FullName
java -jar $C2cpgJar (Join-Path $Here "exemplo.c") --output $Cpg
if (-not (Test-Path $Cpg)) {
    Write-Error "CPG was not created"
    exit 1
}

# Step 2 - export: one .dot file per function, for each representation.
Write-Host "`n[2/3] joern-export" -ForegroundColor Cyan
foreach ($repr in "ast", "cfg", "pdg", "cpg14") {
    & (Join-Path $JoernDir "joern-export.bat") $Cpg --repr $repr --out (Join-Path $Out $repr)
    $n = (Get-ChildItem (Join-Path $Out $repr) -Filter *.dot).Count
    Write-Host "  $repr -> $n .dot files"
}

# Step 3 - query the CPG with a Joern script.
Write-Host "`n[3/3] joern --script consultas.sc" -ForegroundColor Cyan
& (Join-Path $JoernDir "joern.bat") --script (Join-Path $Here "consultas.sc") --param "cpgFile=$Cpg"
