@echo off
echo Testing LaTeX compilation...
echo Checking if pdflatex is available...

where pdflatex >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pdflatex is not installed or not in PATH
    echo Please install TeX Live, MiKTeX, or another LaTeX distribution
    pause
    exit /b 1
)

echo pdflatex found. Attempting compilation...
pdflatex -interaction=nonstopmode QFLARE_IEEE_Paper.tex

if exist QFLARE_IEEE_Paper.pdf (
    echo SUCCESS: PDF generated successfully!
    echo File: QFLARE_IEEE_Paper.pdf
) else (
    echo ERROR: PDF was not generated. Check for compilation errors above.
)

pause