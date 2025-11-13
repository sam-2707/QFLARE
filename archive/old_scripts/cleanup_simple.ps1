# QFLARE Cleanup Script - Simple Direct Approach
Write-Host "Starting QFLARE Project Cleanup..." -ForegroundColor Cyan
Write-Host ""

$removed = 0
$errors = 0

# Phase 1: LaTeX artifacts
Write-Host "Phase 1: Removing LaTeX build artifacts..." -ForegroundColor Yellow
Get-ChildItem -Path "." -Include *.aux,*.log,*.out,*.toc,*.lof,*.lot,*.bcf,*.run.xml,*.nlo,*.loa -Recurse -File | ForEach-Object {
    try {
        Remove-Item $_.FullName -Force
        $removed++
        Write-Host "  Removed: $($_.Name)" -ForegroundColor Gray
    } catch {
        $errors++
    }
}

# Phase 2: Python cache
Write-Host ""
Write-Host "Phase 2: Removing Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path "." -Include __pycache__,.pytest_cache -Recurse -Directory | ForEach-Object {
    try {
        Remove-Item $_.FullName -Recurse -Force
        $removed++
        Write-Host "  Removed: $($_.FullName)" -ForegroundColor Gray
    } catch {
        $errors++
    }
}

# Phase 3: Redundant docs in root
Write-Host ""
Write-Host "Phase 3: Removing redundant status reports..." -ForegroundColor Yellow
$redundantFiles = @(
    "ADVANCED_TESTING_COMPLETE.md",
    "BYZANTINE_IMPLEMENTATION_COMPLETE.md",
    "DIFFERENTIAL_PRIVACY_IMPLEMENTATION.md",
    "FL_CONNECTION_FIX.md",
    "FL_DIRECT_API_FIX.md",
    "FL_IMPLEMENTATION_COMPLETE.md",
    "FL_PROXY_FINAL_FIX.md",
    "FL_PROXY_PATH_FIX.md",
    "FL_STATUS_COMPLETE.md",
    "ISSUE_FIX_SUMMARY.md",
    "ISSUE_RESOLUTION_REPORT.md",
    "REAL_ML_IMPLEMENTATION_COMPLETE.md",
    "WEBSOCKET_IMPLEMENTATION_COMPLETE.md",
    "PROJECT_COMPLETION_STATUS.md",
    "FINAL_PROJECT_STATUS.md",
    "QFLARE_PRODUCTION_STATUS.md",
    "INTEGRATION_COMPLETE.md",
    "FRONTEND_IMPLEMENTATION_COMPLETE.md",
    "FL_WEBSITE_IMPLEMENTATION_COMPLETE.md",
    "COMPREHENSIVE_ENHANCEMENT_COMPLETE.md",
    "FL_API_FIX.md",
    "FL_IMPLEMENTATION_SUMMARY.md",
    "FL_VISUAL_ARCHITECTURE.md",
    "IMPROVEMENTS_GUIDE.md",
    "MVP_COMPLETION_SUMMARY.md",
    "MVP_TESTING_GUIDE.md",
    "NEW_FEATURES_OVERVIEW.md",
    "MODERN_UI_UPDATE.md",
    "SECURITY_DASHBOARD_FIX.md",
    "WHAT_WE_BUILT.md"
)

foreach ($file in $redundantFiles) {
    if (Test-Path $file) {
        try {
            Remove-Item $file -Force
            $removed++
            Write-Host "  Removed: $file" -ForegroundColor Gray
        } catch {
            $errors++
        }
    }
}

# Phase 4: Old scripts
Write-Host ""
Write-Host "Phase 4: Removing old/temporary scripts..." -ForegroundColor Yellow
$oldScripts = @(
    "advanced_testing_suite.py",
    "compile_test.bat",
    "comprehensive_demo.py",
    "performance_dashboard.py",
    "qflare_complete_demo.py",
    "qflare_simple_demo.py",
    "quick_demo.py",
    "run_backend_simple.py",
    "run_qflare.py",
    "run_system.bat",
    "run_validation_suite.py",
    "secure_storage_architecture.py",
    "start_clean_mvp.py",
    "start_frontend.bat",
    "start_qflare.bat",
    "start_qflare.ps1",
    "start_qflare_fixed.bat",
    "start_qflare_simple.py",
    "storage_status_tool.py",
    "test_integration.py",
    "test_real_ml_integration.py",
    "test_websocket_integration.py",
    "verify_fixes.py",
    "simple_diagram_viewer.py",
    "view_diagrams.py",
    "generate_clean_diagrams.py",
    "generate_diagrams.py",
    "generate_presentations.py",
    "qflare_mermaid_diagrams.py"
)

foreach ($script in $oldScripts) {
    if (Test-Path $script) {
        try {
            Remove-Item $script -Force
            $removed++
            Write-Host "  Removed: $script" -ForegroundColor Gray
        } catch {
            $errors++
        }
    }
}

# Phase 5: Old docs
Write-Host ""
Write-Host "Phase 5: Removing old documentation files..." -ForegroundColor Yellow
$oldDocs = @(
    "api_test.html",
    "author.tex",
    "CLEANUP_PLAN.md",
    "QFLARE_IEEE_Paper_Clean.tex",
    "QFLARE_Mathematical_Proofs.tex",
    "requirements.secure.txt",
    "SECURE_STORAGE_STRATEGY.md",
    "system_status_report.json",
    "texput.log",
    "CLEAN_MVP_README.md",
    "QUICK_START_GUIDE.md"
)

foreach ($doc in $oldDocs) {
    if (Test-Path $doc) {
        try {
            Remove-Item $doc -Force
            $removed++
            Write-Host "  Removed: $doc" -ForegroundColor Gray
        } catch {
            $errors++
        }
    }
}

# Phase 6: Log files
Write-Host ""
Write-Host "Phase 6: Removing log files..." -ForegroundColor Yellow
$logFiles = @(
    "qflare_integration.log",
    "qflare_startup.log"
)

foreach ($log in $logFiles) {
    if (Test-Path $log) {
        try {
            Remove-Item $log -Force
            $removed++
            Write-Host "  Removed: $log" -ForegroundColor Gray
        } catch {
            $errors++
        }
    }
}

# Phase 7: Old technical reports
Write-Host ""
Write-Host "Phase 7: Removing old technical reports..." -ForegroundColor Yellow
if (Test-Path "QFLARE_Technical_Report.tex") {
    Remove-Item "QFLARE_Technical_Report.tex" -Force
    $removed++
    Write-Host "  Removed: QFLARE_Technical_Report.tex" -ForegroundColor Gray
}
if (Test-Path "QFLARE_Technical_Report.pdf") {
    Remove-Item "QFLARE_Technical_Report.pdf" -Force
    $removed++
    Write-Host "  Removed: QFLARE_Technical_Report.pdf" -ForegroundColor Gray
}

# Phase 8: Uploaded example files
Write-Host ""
Write-Host "Phase 8: Removing uploaded example files..." -ForegroundColor Yellow
if (Test-Path "CYS_endsem[1].pdf") {
    Remove-Item "CYS_endsem[1].pdf" -Force
    $removed++
    Write-Host "  Removed: CYS_endsem[1].pdf" -ForegroundColor Gray
}
if (Test-Path "CYS endsem.pptx") {
    Remove-Item "CYS endsem.pptx" -Force
    $removed++
    Write-Host "  Removed: CYS endsem.pptx" -ForegroundColor Gray
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Cleanup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  Files Removed: $removed"
Write-Host "  Errors: $errors"
Write-Host ""
Write-Host "Kept Important Files:" -ForegroundColor Green
Write-Host "  - Source code (backend/, frontend/, server/, etc.)"
Write-Host "  - Configuration files (.env, config/, docker/)"
Write-Host "  - Data and models (data/, models/, benchmark_results/)"
Write-Host "  - Documentation (docs/, README.md)"
Write-Host "  - Latest report (QFLARE_Performance_Report.*)"
Write-Host "  - Paper sources (paper/)"
Write-Host "  - Tests (tests/)"
Write-Host "  - Build scripts (build/, scripts/)"
Write-Host ""
Write-Host "Project Structure Optimized!" -ForegroundColor Cyan
