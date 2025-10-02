# PowerShell script to compile QFLARE IEEE Paper
# Usage: .\compile_paper.ps1

Write-Host "Compiling QFLARE IEEE Paper..." -ForegroundColor Green
Write-Host ""

# Check if pdflatex is available
if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Write-Host "Trying main.tex first..." -ForegroundColor Yellow
    
    # First attempt with main.tex
    $result = & pdflatex -interaction=nonstopmode main.tex 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "First pass completed successfully" -ForegroundColor Green
        Write-Host "Running second pass for references..." -ForegroundColor Yellow
        
        $result = & pdflatex -interaction=nonstopmode main.tex 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Compilation successful! Check main.pdf" -ForegroundColor Green
        } else {
            Write-Host "Second pass failed - check main.log for details" -ForegroundColor Red
        }
    } else {
        Write-Host "First pass failed - trying simple version..." -ForegroundColor Yellow
        
        $result = & pdflatex -interaction=nonstopmode QFLARE_IEEE_Paper_Simple.tex 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Simple version compiled successfully! Check QFLARE_IEEE_Paper_Simple.pdf" -ForegroundColor Green
        } else {
            Write-Host "Both attempts failed - check log files for details" -ForegroundColor Red
            Write-Host "Error output:" -ForegroundColor Red
            Write-Host $result
        }
    }
} else {
    Write-Host "pdflatex not found in PATH. Please install LaTeX or add it to your PATH." -ForegroundColor Red
    Write-Host "You can try online LaTeX compilers like Overleaf with the files." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")