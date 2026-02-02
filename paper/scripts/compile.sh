#!/bin/bash
# LaTeX compilation script
# Runs pdflatex + bibtex cycle for complete document compilation

set -e  # Exit on error

# Get script directory and paper directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"

echo "=========================================="
echo "Compiling LaTeX document..."
echo "=========================================="

# First pass: pdflatex
echo "Running pdflatex (pass 1/3)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    echo "Error: pdflatex failed. Check main.log for details."
    exit 1
}

# Run bibtex
if [ -f "bibliography/references.bib" ]; then
    echo "Running bibtex..."
    bibtex main > /dev/null 2>&1 || {
        echo "Warning: bibtex failed. Continuing without bibliography."
    }
else
    echo "Warning: bibliography/references.bib not found. Skipping bibliography."
fi

# Second pass: pdflatex
echo "Running pdflatex (pass 2/3)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    echo "Error: pdflatex failed. Check main.log for details."
    exit 1
}

# Third pass: pdflatex (for cross-references)
echo "Running pdflatex (pass 3/3)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    echo "Error: pdflatex failed. Check main.log for details."
    exit 1
}

echo "=========================================="
if [ -f "main.pdf" ]; then
    echo "✓ Compilation successful: main.pdf"
    echo "=========================================="
else
    echo "✗ Error: main.pdf not generated"
    echo "=========================================="
    exit 1
fi
