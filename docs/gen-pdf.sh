#!/usr/bin/env bash
set -eEuo pipefail
# generate pdf with pandoc/xelatex from markdown files with eisvogel
# https://github.com/Wandmalfarbe/pandoc-latex-template

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$PWD:/data" \
  pandoc/extra \
    --from=markdown \
    --pdf-engine=xelatex \
    --template=eisvogel \
    --filter pandoc-latex-environment \
    --listings \
    --highlight-style kate \
    -o "${1%.*}.pdf" \
    "$1"
