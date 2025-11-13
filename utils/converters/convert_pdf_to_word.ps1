# QFLARE PDF to Word Converter
# Converts QFLARE_Final_Report.pdf to QFLARE_Final_Report.docx

Write-Host "QFLARE PDF to Word Converter" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

$pdfPath = "D:\QFLARE_Project_Structure\QFLARE_Final_Report.pdf"
$wordPath = "D:\QFLARE_Project_Structure\QFLARE_Final_Report.docx"

# Check if PDF exists
if (-not (Test-Path $pdfPath)) {
    Write-Host "ERROR: PDF file not found at: $pdfPath" -ForegroundColor Red
    exit 1
}

Write-Host "PDF file found: $pdfPath" -ForegroundColor Green
Write-Host "Output path: $wordPath" -ForegroundColor Green
Write-Host ""

# Method 1: Using Microsoft Word (if installed)
Write-Host "Attempting Method 1: Microsoft Word Automation..." -ForegroundColor Yellow

try {
    # Create Word Application object
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    
    Write-Host "Opening PDF in Word..." -ForegroundColor Yellow
    
    # Open the PDF file
    $doc = $word.Documents.Open($pdfPath, $false, $true)
    
    Write-Host "Converting to Word format..." -ForegroundColor Yellow
    
    # Save as Word document (16 = wdFormatDocumentDefault for .docx)
    $doc.SaveAs([ref]$wordPath, [ref]16)
    
    # Close document and quit Word
    $doc.Close()
    $word.Quit()
    
    # Release COM objects
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
    
    Write-Host ""
    Write-Host "SUCCESS! Word document created:" -ForegroundColor Green
    Write-Host "  $wordPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "File size: $((Get-Item $wordPath).Length / 1MB) MB" -ForegroundColor Cyan
    
    # Open the Word document
    Write-Host ""
    $response = Read-Host "Would you like to open the Word document now? (Y/N)"
    if ($response -eq 'Y' -or $response -eq 'y') {
        Start-Process $wordPath
    }
    
    exit 0
}
catch {
    Write-Host ""
    Write-Host "Method 1 failed. Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible reasons:" -ForegroundColor Yellow
    Write-Host "  1. Microsoft Word is not installed" -ForegroundColor Yellow
    Write-Host "  2. Word doesn't have permission to open PDFs" -ForegroundColor Yellow
    Write-Host "  3. PDF file is corrupted or locked" -ForegroundColor Yellow
    Write-Host ""
}

# Method 2: Suggest online tools
Write-Host "Alternative Methods:" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Method 2: Online Converters (Most Reliable)" -ForegroundColor Yellow
Write-Host "  1. Adobe Online: https://www.adobe.com/acrobat/online/pdf-to-word.html" -ForegroundColor White
Write-Host "  2. SmallPDF: https://smallpdf.com/pdf-to-word" -ForegroundColor White
Write-Host "  3. ILovePDF: https://www.ilovepdf.com/pdf_to_word" -ForegroundColor White
Write-Host "  4. Zamzar: https://www.zamzar.com/convert/pdf-to-docx/" -ForegroundColor White
Write-Host ""

Write-Host "Method 3: Desktop Software" -ForegroundColor Yellow
Write-Host "  1. Adobe Acrobat Pro DC (Paid)" -ForegroundColor White
Write-Host "  2. LibreOffice (Free): https://www.libreoffice.org/" -ForegroundColor White
Write-Host "  3. WPS Office (Free): https://www.wps.com/" -ForegroundColor White
Write-Host ""

Write-Host "Method 4: Python Package (pdf2docx)" -ForegroundColor Yellow
Write-Host "  Run these commands:" -ForegroundColor White
Write-Host "    pip install pdf2docx" -ForegroundColor Gray
Write-Host "    python -m pdf2docx convert 'QFLARE_Final_Report.pdf' 'QFLARE_Final_Report.docx'" -ForegroundColor Gray
Write-Host ""

# Method 3: Try to install and use pdf2docx if Python is available
Write-Host "Attempting Method 3: Python pdf2docx..." -ForegroundColor Yellow
Write-Host ""

try {
    # Check if Python is installed
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Python found: $pythonVersion" -ForegroundColor Green
        Write-Host ""
        
        $installChoice = Read-Host "Would you like to install pdf2docx and convert now? (Y/N)"
        if ($installChoice -eq 'Y' -or $installChoice -eq 'y') {
            Write-Host ""
            Write-Host "Installing pdf2docx package..." -ForegroundColor Yellow
            python -m pip install pdf2docx
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-Host "Converting PDF to Word using pdf2docx..." -ForegroundColor Yellow
                python -m pdf2docx convert "$pdfPath" "$wordPath"
                
                if ($LASTEXITCODE -eq 0 -and (Test-Path $wordPath)) {
                    Write-Host ""
                    Write-Host "SUCCESS! Word document created using pdf2docx:" -ForegroundColor Green
                    Write-Host "  $wordPath" -ForegroundColor Cyan
                    Write-Host ""
                    Write-Host "File size: $((Get-Item $wordPath).Length / 1MB) MB" -ForegroundColor Cyan
                    
                    # Open the Word document
                    Write-Host ""
                    $response = Read-Host "Would you like to open the Word document now? (Y/N)"
                    if ($response -eq 'Y' -or $response -eq 'y') {
                        Start-Process $wordPath
                    }
                    exit 0
                }
            }
        }
    }
}
catch {
    Write-Host "Python method not available: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "RECOMMENDATION:" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan
Write-Host "The easiest method is to use an online converter:" -ForegroundColor White
Write-Host ""
Write-Host "1. Go to: https://www.ilovepdf.com/pdf_to_word" -ForegroundColor Yellow
Write-Host "2. Upload: QFLARE_Final_Report.pdf" -ForegroundColor Yellow
Write-Host "3. Click 'Convert to Word'" -ForegroundColor Yellow
Write-Host "4. Download the .docx file" -ForegroundColor Yellow
Write-Host ""
Write-Host "This preserves formatting, tables, and images better than most methods." -ForegroundColor Green
Write-Host ""
