# QFLARE Project Cleanup & Organization Plan

**Date:** November 13, 2025  
**Status:** Analysis Complete - Ready for Cleanup  
**Project Root:** `d:\QFLARE_Project_Structure`

---

## 📊 Current Project Analysis

### **Total Files:** 920+ files
### **Directory Structure:** 30+ folders
### **Issues Identified:** 
- ❌ Loose files in root directory (PDFs, images, scripts)
- ❌ LaTeX auxiliary files (.aux, .log, .toc, .lof, .lot, .out) in root
- ❌ Duplicate/temporary scripts (cleanup_files.ps1, cleanup_simple.ps1)
- ❌ Mixed content (source PDFs, reports, scripts) not organized
- ❌ No clear separation between source materials and outputs

---

## 🎯 Cleanup Goals

1. ✅ **Organize root directory** - Move files to appropriate subdirectories
2. ✅ **Clean LaTeX artifacts** - Remove/move auxiliary files
3. ✅ **Archive source materials** - Move PDFs to archive/
4. ✅ **Consolidate scripts** - Organize utility scripts
5. ✅ **Create clear structure** - Separate inputs, outputs, documentation

---

## 📁 Proposed Directory Structure

```
d:\QFLARE_Project_Structure\
│
├── 📄 README.md                          # Main project README (KEEP)
├── 📄 LICENSE                            # License file (KEEP)
├── 📄 .gitignore                         # Git ignore rules (KEEP)
├── 📄 .env.example                       # Environment template (KEEP)
├── 📄 alembic.ini                        # Database migrations (KEEP)
├── 📄 docker-compose.prod.yml            # Production Docker (KEEP)
│
├── 📂 archive/                           # ✅ ALREADY EXISTS
│   ├── source_materials/                 # 🆕 CREATE - Original PDFs
│   │   ├── Cyber (2).pdf
│   │   ├── CYS_endsem[1].pdf
│   │   ├── CYS_Report_NeuroEdge-1.pdf
│   │   └── QFLARE_Performance_Report (1)[1].pdf
│   ├── images/                           # 🆕 CREATE - Source images
│   │   └── WhatsApp Image 2025-11-09 at 13.20.20_fec7f139.jpg
│   └── old_scripts/                      # 🆕 CREATE - Deprecated scripts
│       ├── cleanup_files.ps1
│       ├── cleanup_simple.ps1
│       └── exp11.py
│
├── 📂 backend/                           # ✅ KEEP AS-IS
│   ├── professional_backend.py
│   ├── simple_backend.py
│   ├── fl_api_extensions.py
│   ├── requirements.txt
│   └── [other backend files]
│
├── 📂 frontend/                          # ✅ KEEP AS-IS
│   └── qflare-ui/
│
├── 📂 server/                            # ✅ KEEP AS-IS
│   └── [server implementation]
│
├── 📂 edge_node/                         # ✅ KEEP AS-IS
│   └── [edge node implementation]
│
├── 📂 security/                          # ✅ KEEP AS-IS
│   ├── auth_system.py
│   ├── security_scanner.py
│   └── [other security modules]
│
├── 📂 database/                          # ✅ KEEP AS-IS
│   ├── models.py
│   ├── services.py
│   └── connection.py
│
├── 📂 common/                            # ✅ KEEP AS-IS
│   ├── __init__.py
│   └── error_handling.py
│
├── 📂 config/                            # ✅ KEEP AS-IS
│   └── global_config.yaml
│
├── 📂 data/                              # ✅ KEEP AS-IS
│   ├── keys/
│   ├── logs/
│   ├── MNIST/
│   ├── models/
│   └── updates/
│
├── 📂 models/                            # ✅ KEEP AS-IS
│   └── [ML models]
│
├── 📂 tests/                             # ✅ KEEP AS-IS
│   ├── test_integration.py
│   ├── test_fl_training.py
│   └── [all test files]
│
├── 📂 scripts/                           # ✅ KEEP AS-IS
│   ├── quick_start.py
│   ├── deploy.py
│   ├── demo_complete_flow.py
│   ├── README.md
│   └── [utility scripts]
│
├── 📂 docker/                            # ✅ KEEP AS-IS
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   └── [Dockerfiles]
│
├── 📂 k8s/                               # ✅ KEEP AS-IS
│   └── [Kubernetes configs]
│
├── 📂 monitoring/                        # ✅ KEEP AS-IS
│   └── [monitoring setup]
│
├── 📂 enclaves/                          # ✅ KEEP AS-IS
│   └── [SGX enclaves]
│
├── 📂 liboqs/                            # ✅ KEEP AS-IS
│   └── [PQC library]
│
├── 📂 liboqs-python/                     # ✅ KEEP AS-IS
│   └── [Python bindings]
│
├── 📂 alembic/                           # ✅ KEEP AS-IS
│   └── [Database migrations]
│
├── 📂 qflare-env/                        # ✅ KEEP AS-IS (Python venv)
│   └── [Virtual environment]
│
├── 📂 build/                             # ✅ KEEP AS-IS
│   └── [Build artifacts]
│
├── 📂 experiments/                       # ✅ KEEP AS-IS
│   └── [Experimental code]
│
├── 📂 benchmark_results/                 # ✅ KEEP AS-IS
│   └── [Performance benchmarks]
│
├── 📂 auth/                              # ✅ KEEP AS-IS
│   └── [Authentication modules]
│
├── 📂 assets/                            # ✅ KEEP AS-IS
│   └── [Static assets]
│
├── 📂 src/                               # ✅ KEEP AS-IS
│   ├── security/
│   └── tools/
│
├── 📂 docs/                              # 🔧 REORGANIZE
│   ├── reports/                          # 🆕 CREATE - All reports
│   │   ├── latex/                        # 🆕 CREATE - LaTeX sources
│   │   │   ├── QFLARE_Final_Report.tex
│   │   │   ├── QFLARE_Final_Report.aux
│   │   │   ├── QFLARE_Final_Report.log
│   │   │   ├── QFLARE_Final_Report.toc
│   │   │   ├── QFLARE_Final_Report.lof
│   │   │   ├── QFLARE_Final_Report.lot
│   │   │   ├── QFLARE_Final_Report.out
│   │   │   ├── QFLARE_Performance_Report.tex
│   │   │   ├── QFLARE_Performance_Report.aux
│   │   │   ├── QFLARE_Performance_Report.log
│   │   │   ├── QFLARE_Performance_Report.toc
│   │   │   ├── QFLARE_Performance_Report.lof
│   │   │   ├── QFLARE_Performance_Report.lot
│   │   │   └── QFLARE_Performance_Report.out
│   │   ├── pdf/                          # 🆕 CREATE - Final PDFs
│   │   │   ├── QFLARE_Final_Report.pdf
│   │   │   └── QFLARE_Performance_Report.pdf
│   │   ├── docx/                         # 🆕 CREATE - Word documents
│   │   │   └── QFLARE_Final_Report.docx
│   │   └── markdown/                     # 🆕 CREATE - Markdown reports
│   │       ├── QFLARE_Performance_Report.md
│   │       ├── QFLARE_PPT_RESULTS.md
│   │       └── QFLARE_PPT_GRAPHS_GUIDE.md
│   ├── guides/                           # 🆕 CREATE - User guides
│   │   ├── QUICK_ACTION_GUIDE.md
│   │   ├── REPORT_STRUCTURE_GUIDE.md
│   │   └── REPORT_UPDATES_SUMMARY.md
│   ├── summaries/                        # 🆕 CREATE - Status docs
│   │   ├── LATEX_ORGANIZATION_SUMMARY.md
│   │   ├── CLEANUP_COMPLETE_SUMMARY.md
│   │   └── PROJECT_CLEANUP_PLAN.md
│   ├── diagrams/                         # 🆕 CREATE - Mermaid diagrams
│   │   └── QFLARE_Mermaid_Diagrams.md
│   ├── figures/                          # 🆕 CREATE - Generated figures
│   │   └── [PNG exports from Mermaid - TO BE ADDED]
│   ├── api/                              # ✅ EXISTING
│   │   ├── api_docs.md
│   │   └── api_documentation.md
│   └── [other existing docs]
│
├── 📂 paper/                             # ✅ KEEP AS-IS
│   ├── main.tex
│   ├── main.pdf
│   ├── QFLARE_IEEE_Paper.tex
│   ├── QFLARE_IEEE_Paper.pdf
│   ├── figures/
│   │   ├── accuracy_comparison.pdf
│   │   ├── performance_overhead.pdf
│   │   ├── qflare_architecture.pdf
│   │   ├── scalability_analysis.pdf
│   │   └── security_radar.pdf
│   ├── compile_paper.ps1
│   ├── generate_figures.py
│   └── README.md
│
└── 📂 utils/                             # 🆕 CREATE - Utility scripts
    ├── converters/                       # 🆕 CREATE - Conversion tools
    │   ├── convert_pdf_to_word.ps1
    │   └── pdf_to_word_converter.py
    └── [other utilities]
```

---

## 🔧 Cleanup Actions

### **Phase 1: Create New Directory Structure (1 minute)**

```powershell
# Create new directories
mkdir docs\reports\latex -Force
mkdir docs\reports\pdf -Force
mkdir docs\reports\docx -Force
mkdir docs\reports\markdown -Force
mkdir docs\guides -Force
mkdir docs\summaries -Force
mkdir docs\diagrams -Force
mkdir docs\figures -Force
mkdir archive\source_materials -Force
mkdir archive\images -Force
mkdir archive\old_scripts -Force
mkdir utils\converters -Force
```

### **Phase 2: Move LaTeX Files (2 minutes)**

```powershell
# Move LaTeX source files
Move-Item "QFLARE_Final_Report.tex" "docs\reports\latex\" -Force
Move-Item "QFLARE_Final_Report.aux" "docs\reports\latex\" -Force
Move-Item "QFLARE_Final_Report.log" "docs\reports\latex\" -Force
Move-Item "QFLARE_Final_Report.toc" "docs\reports\latex\" -Force
Move-Item "QFLARE_Final_Report.lof" "docs\reports\latex\" -Force
Move-Item "QFLARE_Final_Report.lot" "docs\reports\latex\" -Force
Move-Item "QFLARE_Final_Report.out" "docs\reports\latex\" -Force

Move-Item "QFLARE_Performance_Report.tex" "docs\reports\latex\" -Force
Move-Item "QFLARE_Performance_Report.aux" "docs\reports\latex\" -Force
Move-Item "QFLARE_Performance_Report.log" "docs\reports\latex\" -Force
Move-Item "QFLARE_Performance_Report.toc" "docs\reports\latex\" -Force
Move-Item "QFLARE_Performance_Report.lof" "docs\reports\latex\" -Force
Move-Item "QFLARE_Performance_Report.lot" "docs\reports\latex\" -Force
Move-Item "QFLARE_Performance_Report.out" "docs\reports\latex\" -Force

# Move PDF outputs
Move-Item "QFLARE_Final_Report.pdf" "docs\reports\pdf\" -Force
Move-Item "QFLARE_Performance_Report.pdf" "docs\reports\pdf\" -Force

# Move Word document
Move-Item "QFLARE_Final_Report.docx" "docs\reports\docx\" -Force

# Move Markdown reports
Move-Item "QFLARE_Performance_Report.md" "docs\reports\markdown\" -Force
Move-Item "QFLARE_PPT_RESULTS.md" "docs\reports\markdown\" -Force
Move-Item "QFLARE_PPT_GRAPHS_GUIDE.md" "docs\reports\markdown\" -Force
```

### **Phase 3: Move Documentation Files (1 minute)**

```powershell
# Move guide documents
Move-Item "docs\QUICK_ACTION_GUIDE.md" "docs\guides\" -Force -ErrorAction SilentlyContinue
Move-Item "REPORT_STRUCTURE_GUIDE.md" "docs\guides\" -Force -ErrorAction SilentlyContinue
Move-Item "REPORT_UPDATES_SUMMARY.md" "docs\guides\" -Force -ErrorAction SilentlyContinue

# Move summary documents
Move-Item "docs\LATEX_ORGANIZATION_SUMMARY.md" "docs\summaries\" -Force -ErrorAction SilentlyContinue
Move-Item "CLEANUP_COMPLETE_SUMMARY.md" "docs\summaries\" -Force -ErrorAction SilentlyContinue

# Move Mermaid diagrams
Move-Item "docs\QFLARE_Mermaid_Diagrams.md" "docs\diagrams\" -Force -ErrorAction SilentlyContinue
```

### **Phase 4: Archive Source Materials (1 minute)**

```powershell
# Move source PDFs to archive
Move-Item "Cyber (2).pdf" "archive\source_materials\" -Force
Move-Item "CYS_endsem[1].pdf" "archive\source_materials\" -Force
Move-Item "CYS_Report_NeuroEdge-1.pdf" "archive\source_materials\" -Force
Move-Item "QFLARE_Performance_Report (1)[1].pdf" "archive\source_materials\" -Force

# Move images to archive
Move-Item "WhatsApp Image 2025-11-09 at 13.20.20_fec7f139.jpg" "archive\images\" -Force

# Move old scripts to archive
Move-Item "cleanup_files.ps1" "archive\old_scripts\" -Force
Move-Item "cleanup_simple.ps1" "archive\old_scripts\" -Force
Move-Item "exp11.py" "archive\old_scripts\" -Force
```

### **Phase 5: Organize Utility Scripts (30 seconds)**

```powershell
# Move converter scripts
Move-Item "convert_pdf_to_word.ps1" "utils\converters\" -Force
Move-Item "pdf_to_word_converter.py" "utils\converters\" -Force
```

### **Phase 6: Create Index/README Files (2 minutes)**

```powershell
# Will create comprehensive README files for each section
# (Detailed content to be added)
```

---

## ✅ Benefits After Cleanup

### **Before Cleanup:**
```
d:\QFLARE_Project_Structure\
├── 50+ loose files in root 😰
├── LaTeX artifacts mixed with source 😰
├── PDFs scattered everywhere 😰
├── No clear organization 😰
└── Hard to find anything 😰
```

### **After Cleanup:**
```
d:\QFLARE_Project_Structure\
├── Clean root (only essential config files) ✅
├── docs/
│   ├── reports/ (all LaTeX, PDF, DOCX organized) ✅
│   ├── guides/ (all user guides) ✅
│   ├── summaries/ (all status docs) ✅
│   ├── diagrams/ (Mermaid sources) ✅
│   └── figures/ (generated PNGs) ✅
├── archive/ (source materials, old scripts) ✅
├── utils/ (converter tools) ✅
└── [all source code folders organized] ✅
```

---

## 📋 Verification Checklist

After running cleanup, verify:

- [ ] **Root directory clean** - Only README, LICENSE, .env, docker-compose in root
- [ ] **LaTeX files organized** - All .tex, .aux, .log, etc. in docs/reports/latex/
- [ ] **PDFs in pdf folder** - Final reports in docs/reports/pdf/
- [ ] **Documentation organized** - Guides, summaries in proper subfolders
- [ ] **Source materials archived** - All input PDFs in archive/source_materials/
- [ ] **Scripts organized** - Converters in utils/converters/
- [ ] **No broken imports** - All code still runs (check main modules)
- [ ] **Git status clean** - Run `git status` to see changes

---

## 🚀 Execution Time Estimate

| Phase | Task | Time |
|-------|------|------|
| 1 | Create directories | 1 min |
| 2 | Move LaTeX files | 2 min |
| 3 | Move documentation | 1 min |
| 4 | Archive materials | 1 min |
| 5 | Organize utilities | 30 sec |
| 6 | Create READMEs | 2 min |
| 7 | Verification | 1 min |
| **Total** | **Complete cleanup** | **~8-10 minutes** |

---

## ⚠️ Important Notes

1. **Backup First:** Consider creating a backup before running cleanup
2. **Virtual Environment:** qflare-env/ folder is large but necessary (keep as-is)
3. **Git Tracking:** Some files may need `.gitignore` updates
4. **Import Paths:** Check if any scripts reference moved files
5. **Database File:** `qflare_mvp.db` should stay in root for now

---

## 🎯 Next Steps

1. **Review this plan** - Confirm directory structure is acceptable
2. **Run cleanup script** - Execute PowerShell commands
3. **Verify functionality** - Test that main scripts still work
4. **Update documentation** - Add README files to new folders
5. **Commit changes** - Git commit with message "Organize project structure"

---

**Ready to execute?** Reply "yes" to run the automated cleanup script.
