# Run Main.py with the project's virtual environment
& "$PSScriptRoot\..\.venv\Scripts\python.exe" "$PSScriptRoot\Main.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
