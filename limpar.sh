#!/bin/bash
find dissertacao -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.loq" -o -name "*.synctex.gz" -o -name "*.blg" -o -name "*.fdb_latexmk" -o -name "*.fls" -o -name "*.idx" -o -name "*.ilg" -o -name "*.ind" -o -name "*.lof" -o -name "*.log" -o -name "*.lot" -o -name "*.toc" \) -exec rm {} +
find . -type d -name "__pycache__" -exec rm -r {} +