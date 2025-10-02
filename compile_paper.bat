@echo off
echo Compiling QFLARE IEEE Paper...
pdflatex -interaction=nonstopmode QFLARE_IEEE_Paper.tex
if %ERRORLEVEL% EQU 0 (
    echo First pass completed successfully
    echo Running second pass for references...
    pdflatex -interaction=nonstopmode QFLARE_IEEE_Paper.tex
    if %ERRORLEVEL% EQU 0 (
        echo Compilation successful! Check QFLARE_IEEE_Paper.pdf
    ) else (
        echo Second pass failed - check log file
    )
) else (
    echo First pass failed - check log file
)
pause