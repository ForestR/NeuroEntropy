#!/bin/bash
# Clean LaTeX auxiliary files

# Get script directory and paper directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"

echo "Cleaning LaTeX auxiliary files..."

# Remove auxiliary files
rm -f *.aux *.log *.out *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls *.toc *.lof *.lot

# Remove backup files
rm -f *~ *.backup

echo "Done. Cleaned auxiliary files."
echo "Note: main.pdf is preserved. Remove it manually if needed."
