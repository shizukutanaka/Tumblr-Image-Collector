#!/bin/bash
# Move speculative/duplicate documentation to archived

# Create archive directory if not exists
mkdir -p docs/archived

# Move speculative improvement documents
mv -f COMPREHENSIVE_IMPROVEMENTS_500.md docs/archived/ 2>/dev/null
mv -f COMMERCIAL_READINESS_SUMMARY.md docs/archived/ 2>/dev/null
mv -f ENHANCED_COLLECTORS_IMPROVEMENTS.md docs/archived/ 2>/dev/null
mv -f FINAL_IMPROVEMENTS_COMPLETE.md docs/archived/ 2>/dev/null
mv -f IMPROVEMENTS_2025.md docs/archived/ 2>/dev/null
mv -f IMPROVEMENTS_SUMMARY.md docs/archived/ 2>/dev/null
mv -f IMPROVEMENT_BACKLOG_STRIPE.md docs/archived/ 2>/dev/null
mv -f QUICKSTART_2025.md docs/archived/ 2>/dev/null
mv -f QUICK_START_ENHANCED.md docs/archived/ 2>/dev/null
mv -f README_ULTIMATE.md docs/archived/ 2>/dev/null
mv -f improvements_analysis.md docs/archived/ 2>/dev/null

# Move legal documents (optional)
mv -f EULA.md docs/archived/ 2>/dev/null
mv -f TERMS_OF_SERVICE.md docs/archived/ 2>/dev/null

# Keep only primary documentation
echo "Documentation archival complete"
ls -la docs/archived/ | wc -l
