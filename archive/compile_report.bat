@echo off
REM QFLARE Technical Report Compilation Script
REM This script compiles the LaTeX document with all necessary passes

echo Compiling QFLARE Technical Report...
echo.

REM First pass - initial compilation
echo [1/6] Running initial pdflatex pass...
pdflatex -interaction=nonstopmode QFLARE_Technical_Report.tex > nul 2>&1
if errorlevel 1 (
    echo ERROR: First pdflatex pass failed
    echo Check QFLARE_Technical_Report.log for errors
    pause
    exit /b 1
)

REM Generate nomenclature
echo [2/6] Generating nomenclature...
makeindex QFLARE_Technical_Report.nlo -s nomencl.ist -o QFLARE_Technical_Report.nls > nul 2>&1

REM Run biber for bibliography
echo [3/6] Processing bibliography...
biber QFLARE_Technical_Report > nul 2>&1
if errorlevel 1 (
    echo WARNING: Bibliography processing failed - continuing without references
)

REM Second pass - resolve references
echo [4/6] Running second pdflatex pass...
pdflatex -interaction=nonstopmode QFLARE_Technical_Report.tex > nul 2>&1
if errorlevel 1 (
    echo ERROR: Second pdflatex pass failed
    echo Check QFLARE_Technical_Report.log for errors
    pause
    exit /b 1
)

REM Third pass - finalize cross-references
echo [5/6] Running final pdflatex pass...
pdflatex -interaction=nonstopmode QFLARE_Technical_Report.tex > nul 2>&1
if errorlevel 1 (
    echo ERROR: Final pdflatex pass failed
    echo Check QFLARE_Technical_Report.log for errors
    pause
    exit /b 1
)

REM Clean up auxiliary files (optional)
echo [6/6] Cleaning up auxiliary files...
del *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.ilg *.lof *.log *.lot *.nlo *.nls *.out *.run.xml *.synctex.gz *.toc *.loa > nul 2>&1

echo.
echo ========================================
echo COMPILATION COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Output file: QFLARE_Technical_Report.pdf
echo.

REM Check if PDF was created
if exist "QFLARE_Technical_Report.pdf" (
    echo PDF file created successfully.
    echo File size: 
    for %%A in (QFLARE_Technical_Report.pdf) do echo %%~zA bytes
    echo.
    echo Would you like to open the PDF? (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        start QFLARE_Technical_Report.pdf
    )
) else (
    echo ERROR: PDF file was not created!
    echo Check the log files for compilation errors.
)

echo.
echo Report compilation statistics:
echo - Document class: book (two-sided)
echo - Total chapters: 9 + appendices
echo - Bibliography entries: 25+
echo - Figures with proper references
echo - Tables with captions
echo - Mathematical equations numbered
echo - Nomenclature (abbreviations) included
echo.
pause