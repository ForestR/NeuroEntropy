#!/bin/bash
# Count words in manuscript (excluding LaTeX commands)

# Get script directory and paper directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"

echo "Counting words in manuscript..."

# Count words in all .tex files, excluding comments and LaTeX commands
# This is a simple approximation - more sophisticated tools exist
total_words=$(find sections -name "*.tex" -exec cat {} \; | \
    grep -v "^%" | \
    sed 's/\\[a-zA-Z]*//g' | \
    sed 's/{[^}]*}//g' | \
    sed 's/\[[^\]]*\]//g' | \
    tr ' ' '\n' | \
    grep -v '^$' | \
    wc -l)

echo "Approximate word count: $total_words"
echo ""
echo "Note: This is an approximation. For accurate counts, use:"
echo "  - detex (if available)"
echo "  - texcount (LaTeX-specific word counter)"
echo "  - Manual counting in compiled PDF"
