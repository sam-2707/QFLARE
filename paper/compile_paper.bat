@echo off
echo Compiling QFLARE IEEE Paper...
echo.
echo Trying main.tex first...
pdflatex -interaction=nonstopmode main.tex
if %ERRORLEVEL% EQU 0 (
    echo First pass completed successfully
    echo Running second pass for references...
    pdflatex -interaction=nonstopmode main.tex
    if %ERRORLEVEL% EQU 0 (
        echo Compilation successful! Check main.pdf
    ) else (
        echo Second pass failed - check main.log for details
    )
) else (
    echo First pass failed - trying simple version...
    pdflatex -interaction=nonstopmode QFLARE_IEEE_Paper_Simple.tex
    if %ERRORLEVEL% EQU 0 (
        echo Simple version compiled successfully! Check QFLARE_IEEE_Paper_Simple.pdf
    ) else (
        echo Both attempts failed - check log files for details
    )
)
pause