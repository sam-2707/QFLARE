# QFLARE Project Cleanup Script
# This script removes unnecessary files while preserving important data

Write-Host "🧹 Starting QFLARE Project Cleanup..." -ForegroundColor Cyan
Write-Host ""

$rootDir = "d:\QFLARE_Project_Structure"
$filesRemoved = 0
$bytesFreed = 0

# Function to safely remove file
function Remove-SafeFile {
    param([string]$path)
    if (Test-Path $path) {
        $size = (Get-Item $path).Length
        Remove-Item $path -Force
        $script:filesRemoved++
        $script:bytesFreed += $size
        Write-Host "✓ Removed: $path" -ForegroundColor Green
    }
}

# Function to safely remove directory
function Remove-SafeDirectory {
    param([string]$path)
    if (Test-Path $path) {
        $size = (Get-ChildItem $path -Recurse | Measure-Object -Property Length -Sum).Sum
        Remove-Item $path -Recurse -Force
        $script:filesRemoved++
        $script:bytesFreed += $size
        Write-Host "✓ Removed: $path" -ForegroundColor Green
    }
}

Write-Host "📝 Phase 1: Removing LaTeX build artifacts..." -ForegroundColor Yellow

# LaTeX build artifacts in root
$latexExtensions = @("*.aux", "*.log", "*.out", "*.toc", "*.lof", "*.lot", "*.bcf", "*.run.xml", "*.nlo", "*.loa")
foreach ($ext in $latexExtensions) {
    Get-ChildItem -Path $rootDir -Filter $ext -Recurse | ForEach-Object {
        Remove-SafeFile $_.FullName
    }
}

Write-Host ""
Write-Host "📝 Phase 2: Removing redundant status reports in root..." -ForegroundColor Yellow

# Redundant status reports (these are duplicated in docs/status-reports/)
$redundantDocs = @(
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
    "QFLARE_PRODUCTION_STATUS.md"
)

foreach ($doc in $redundantDocs) {
    Remove-SafeFile (Join-Path $rootDir $doc)
}

Write-Host ""
Write-Host "📝 Phase 3: Removing old implementation status files..." -ForegroundColor Yellow

$oldStatusFiles = @(
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

foreach ($file in $oldStatusFiles) {
    Remove-SafeFile (Join-Path $rootDir $file)
}

Write-Host ""
Write-Host "📝 Phase 4: Removing old/temporary scripts..." -ForegroundColor Yellow

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
    Remove-SafeFile (Join-Path $rootDir $script)
}

Write-Host ""
Write-Host "📝 Phase 5: Removing old documentation files..." -ForegroundColor Yellow

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
    Remove-SafeFile (Join-Path $rootDir $doc)
}

Write-Host ""
Write-Host "📝 Phase 6: Removing Python cache directories..." -ForegroundColor Yellow

Get-ChildItem -Path $rootDir -Filter "__pycache__" -Directory -Recurse | ForEach-Object {
    Remove-SafeDirectory $_.FullName
}

Get-ChildItem -Path $rootDir -Filter ".pytest_cache" -Directory -Recurse | ForEach-Object {
    Remove-SafeDirectory $_.FullName
}

Write-Host ""
Write-Host "📝 Phase 7: Removing log files..." -ForegroundColor Yellow

$logFiles = @(
    "qflare_integration.log",
    "qflare_startup.log"
)

foreach ($log in $logFiles) {
    Remove-SafeFile (Join-Path $rootDir $log)
}

Write-Host ""
Write-Host "📝 Phase 8: Removing redundant technical reports (keeping latest)..." -ForegroundColor Yellow

# Remove old technical report (we have QFLARE_Performance_Report which is the latest)
Remove-SafeFile (Join-Path $rootDir "QFLARE_Technical_Report.tex")
Remove-SafeFile (Join-Path $rootDir "QFLARE_Technical_Report.pdf")

# Keep only the PPT guide we created, remove intermediate docs
$docsToRemove = @(
    "QFLARE_PPT_GRAPHS_GUIDE.md",  # Can recreate if needed
    "QFLARE_PPT_RESULTS.md"         # Can recreate if needed
)

# Actually, let's keep these as they're useful references
# Commenting them out
# foreach ($doc in $docsToRemove) {
#     Remove-SafeFile (Join-Path $rootDir $doc)
# }

Write-Host ""
Write-Host "📝 Phase 9: Removing old presentation/demo files..." -ForegroundColor Yellow

# Remove uploaded example file (no longer needed)
Remove-SafeFile (Join-Path $rootDir "CYS_endsem[1].pdf")
Remove-SafeFile (Join-Path $rootDir "CYS endsem.pptx")

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🎉 Cleanup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor Yellow
Write-Host "  Files/Folders Removed: $filesRemoved"
Write-Host "  Space Freed: $([math]::Round($bytesFreed / 1MB, 2)) MB"
Write-Host ""
Write-Host "Kept Important Files:" -ForegroundColor Green
Write-Host "  - Source code (backend/, frontend/, server/, edge_node/, etc.)"
Write-Host "  - Configuration files (.env, config/, docker/)"
Write-Host "  - Data and models (data/, models/, benchmark_results/)"
Write-Host "  - Documentation (docs/, README.md)"
Write-Host "  - Latest reports (QFLARE_Performance_Report.*)"
Write-Host "  - Paper sources (paper/)"
Write-Host "  - Tests (tests/)"
Write-Host "  - Build scripts (build/, scripts/)"
Write-Host ""
Write-Host "Project Structure Optimized!" -ForegroundColor Cyan
