QFLARE Paper build instructions

Files of interest:
- main.tex: Primary safe entry point. Includes `QFLARE_IEEE_Paper_Simple.tex` by default.
- QFLARE_IEEE_Paper.tex: Full paper source (cleaned and structure-validated).
- QFLARE_IEEE_Paper_Simple.tex: Simplified version guaranteed to compile on minimal LaTeX installs.
- compile_paper.ps1 / compile_paper.bat: Scripts for compiling on Windows (prefer PowerShell script).
- validate_latex.ps1: Script to perform basic structural validation.

To build (PowerShell):

```powershell
# Validate first
.\validate_latex.ps1 main.tex

# Compile
.\compile_paper.ps1
```

If `pdflatex` is not available, install a TeX distribution (TeX Live or MiKTeX) or upload the files to Overleaf.
