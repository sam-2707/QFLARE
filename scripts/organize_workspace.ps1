<#
.SYNOPSIS
    Organize the workspace by moving paper-related files into a `paper/` folder and
    archiving common LaTeX auxiliary files.

.DESCRIPTION
    This script moves a conservative set of files that are related to the LaTeX
    paper into a new `paper/` directory. It also moves common LaTeX aux files
    (e.g. .aux, .log) into `paper/archive/` to keep the repository root clean.

    The script is intentionally conservative: it only moves the files listed in
    $FilesToMove (paper-related) and common aux files. It will not delete any
    files. All actions are logged to `organize.log` in the repository root.

.PARAMETER Execute
    When provided, the script will actually perform the moves. Without it, the
    script runs in dry-run mode and prints what it would do.

.EXAMPLE
    # Dry run (default)
    .\scripts\organize_workspace.ps1

    # Execute moves
    .\scripts\organize_workspace.ps1 -Execute
#>
param(
    [switch]$Execute
)

Set-StrictMode -Version Latest
Push-Location -Path "$(Split-Path -Path $MyInvocation.MyCommand.Definition -Parent)\.."

$RepoRoot = Get-Location
$LogFile = Join-Path $RepoRoot 'organize.log'

function Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToString('u')
    $line = "$timestamp`t$Message"
    $line | Tee-Object -FilePath $LogFile -Append
}

Log "Starting workspace organization (Execute=$Execute)"

# Directories to create
$PaperDir = Join-Path $RepoRoot 'paper'
$PaperArchive = Join-Path $PaperDir 'archive'
$DocsDir = Join-Path $RepoRoot 'docs'
$BuildDir = Join-Path $RepoRoot 'build'
$DataKeysDir = Join-Path $RepoRoot 'data/keys'

# Function to ensure directory exists
function EnsureDirectory {
    param([string]$Path, [string]$Purpose)
    if (-not (Test-Path $Path)) {
        if ($Execute) {
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
            Log "Created directory: $Path ($Purpose)"
        } else {
            Log "DRY-RUN: Would create directory: $Path ($Purpose)"
        }
    } else {
        Log "Directory already exists: $Path ($Purpose)"
    }
}

# Create all needed directories
EnsureDirectory $PaperDir "LaTeX paper files"
EnsureDirectory $PaperArchive "LaTeX auxiliary files"
EnsureDirectory $DocsDir "Documentation and markdown files"
EnsureDirectory $BuildDir "Build and deployment files"
EnsureDirectory $DataKeysDir "Keys and security tokens"

# Files to organize by category
$PaperFiles = @(
    'main.tex',
    'QFLARE_IEEE_Paper.tex',
    'QFLARE_IEEE_Paper_Simple.tex',
    'README_BUILD.md',
    'compile_paper.ps1',
    'compile_paper.bat',
    'validate_latex.ps1'
)

# Markdown documentation files
$DocsFiles = @(
    'README.md',
    'CLEANUP_SUMMARY.md',
    'ORGANIZATION_COMPLETE.md',
    'PORT_CHANGE_SUMMARY.md',
    'ORGANIZE_SUMMARY.md',
    'PROJECT_STATUS.md',
    'PROJECT_STRUCTURE.md',
    'QFLARE_Security_Summary.txt',
    'quantum_key_overview.md',
    'quantum_key_usage_guide.md',
    'SECURE_KEY_EXCHANGE_GUIDE.md',
    'SECURITY_DEMONSTRATION_GUIDE.md',
    'TROUBLESHOOTING.md'
)

# Build and deployment files
$BuildFiles = @(
    'deploy.bat',
    'deploy.sh',
    'docker-compose.yml',
    'docker-compose.prod.yml',
    'Dockerfile.prod',
    'requirements.txt',
    'requirements.prod.txt',
    'setup.py',
    'start_qflare.py'
)

# Data and key files
$DataFiles = @(
    'admin_master_key.pem',
    'api_test.html',
    'alembic.ini',
    'l.json',
    'LICENSE',
    'qr_code_*.txt',
    'qr_code_*.json',
    'security_token_*.json'
)

# Function to move files to a destination directory
function MoveFiles {
    param([array]$Files, [string]$DestDir, [string]$Category)
    foreach ($f in $Files) {
        # Handle wildcard patterns
        if ($f -like "*`**") {
            $matches = Get-ChildItem -Path $RepoRoot -Filter $f -File -ErrorAction SilentlyContinue
            foreach ($match in $matches) {
                $src = $match.FullName
                $dest = Join-Path $DestDir $match.Name
                if ($Execute) {
                    try {
                        Move-Item -Path $src -Destination $dest -Force -ErrorAction Stop
                        Log "Moved: $($match.Name) -> $Category/$($match.Name)"
                    } catch {
                        Log ("ERROR moving {0}: {1}" -f $match.Name, $_)
                    }
                } else {
                    Log "DRY-RUN: Would move $($match.Name) -> $Category/$($match.Name)"
                }
            }
        } else {
            $src = Join-Path $RepoRoot $f
            if (Test-Path $src) {
                $dest = Join-Path $DestDir $f
                if ($Execute) {
                    try {
                        Move-Item -Path $src -Destination $dest -Force -ErrorAction Stop
                        Log "Moved: $f -> $Category/$f"
                    } catch {
                        Log ("ERROR moving {0}: {1}" -f $f, $_)
                    }
                } else {
                    Log "DRY-RUN: Would move $f -> $Category/$f"
                }
            } else {
                Log "Not found (skipped): $f"
            }
        }
    }
}

# Move files to their respective directories
MoveFiles $PaperFiles $PaperDir "paper"
MoveFiles $DocsFiles $DocsDir "docs"
MoveFiles $BuildFiles $BuildDir "build"

# Move data/key files with wildcard support
$DataMatches = Get-ChildItem -Path $RepoRoot -Filter "qr_code_*" -File -ErrorAction SilentlyContinue
$DataMatches += Get-ChildItem -Path $RepoRoot -Filter "security_token_*" -File -ErrorAction SilentlyContinue
$DataMatches += Get-ChildItem -Path $RepoRoot -Filter "admin_master_key.pem" -File -ErrorAction SilentlyContinue

foreach ($match in $DataMatches) {
    $src = $match.FullName
    $dest = Join-Path $DataKeysDir $match.Name
    if ($Execute) {
        try {
            Move-Item -Path $src -Destination $dest -Force -ErrorAction Stop
            Log "Moved: $($match.Name) -> data/keys/$($match.Name)"
        } catch {
            Log ("ERROR moving {0}: {1}" -f $match.Name, $_)
        }
    } else {
        Log "DRY-RUN: Would move $($match.Name) -> data/keys/$($match.Name)"
    }
}

# Archive common LaTeX auxiliary files from the repo root
$AuxPatterns = @('*.aux','*.log','*.out','*.toc','*.lof','*.lot')
foreach ($pattern in $AuxPatterns) {
    $matches = Get-ChildItem -Path $RepoRoot -Filter $pattern -File -ErrorAction SilentlyContinue
    foreach ($m in $matches) {
        $src = $m.FullName
        $dest = Join-Path $PaperArchive $m.Name
        if ($Execute) {
            try {
                Move-Item -Path $src -Destination $dest -Force -ErrorAction Stop
                Log "Archived: $($m.Name) -> paper/archive/$($m.Name)"
            } catch {
                Log ("ERROR archiving {0}: {1}" -f $($m.Name), $_)
            }
        } else {
            Log "DRY-RUN: Would archive $($m.Name) -> paper/archive/$($m.Name)"
        }
    }
}

Log "Organization complete. Review organize.log for details."

Pop-Location
