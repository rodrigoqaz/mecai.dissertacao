#!/bin/bash
find dissertacao -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.loq" -o -name "*.synctex.gz" \) -exec rm {} +