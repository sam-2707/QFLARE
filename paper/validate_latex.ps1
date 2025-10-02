# LaTeX Validation Script
# Checks for common LaTeX compilation issues

param([string]$filename = "main.tex")

Write-Host "Validating LaTeX file: $filename" -ForegroundColor Green
Write-Host "=" * 50

if (-not (Test-Path $filename)) {
    Write-Host "Error: File $filename not found!" -ForegroundColor Red
    exit 1
}

# Check begin/end balance
$beginCount = (Select-String "\\begin\{" $filename).Count
$endCount = (Select-String "\\end\{" $filename).Count

Write-Host "Environment Balance Check:" -ForegroundColor Yellow
Write-Host "  \begin{} commands: $beginCount"
Write-Host "  \end{} commands: $endCount"

if ($beginCount -eq $endCount) {
    Write-Host "  OK: Environments are balanced" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Environment mismatch detected!" -ForegroundColor Red
}

# Check for common problematic patterns
Write-Host ""
Write-Host "Common Issue Checks:" -ForegroundColor Yellow

# Check for undefined references
$undefRefs = Select-String "\\\\ref\\{" $filename
if ($undefRefs.Count -gt 0) {
    Write-Host "  Warning: Found $($undefRefs.Count) references - ensure targets exist" -ForegroundColor Yellow
} else {
    Write-Host "  OK: No \ref{} commands found" -ForegroundColor Green
}

# Check for citation commands
$citations = Select-String "\\\\cite\\{" $filename
if ($citations.Count -gt 0) {
    Write-Host "  Warning: Found $($citations.Count) citations - ensure bibliography entries exist" -ForegroundColor Yellow
} else {
    Write-Host "  OK: No \cite{} commands found" -ForegroundColor Green
}

# Check for problematic packages
$problematicPackages = @("tikz", "pgfplots", "listings")
foreach ($pkg in $problematicPackages) {
    $packageUse = Select-String "\\\\usepackage.*$pkg" $filename
    if ($packageUse.Count -gt 0) {
        Write-Host "  Warning: Uses potentially problematic package: $pkg" -ForegroundColor Yellow
    }
}

# Check for unbalanced braces (basic check)
$content = Get-Content $filename -Raw
$openBraces = ($content.ToCharArray() | Where-Object { $_ -eq '{' }).Count
$closeBraces = ($content.ToCharArray() | Where-Object { $_ -eq '}' }).Count

Write-Host ""
Write-Host "Brace Balance Check:" -ForegroundColor Yellow
Write-Host "  Open braces: $openBraces"
Write-Host "  Close braces: $closeBraces"

if ($openBraces -eq $closeBraces) {
    Write-Host "  OK: Braces appear balanced" -ForegroundColor Green
} else {
    Write-Host "  Warning: Brace count mismatch - may indicate syntax error" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Validation complete!" -ForegroundColor Green