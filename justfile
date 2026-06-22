set shell := ["bash", "-euo", "pipefail", "-c"]

sync:
    git add .
    git commit
    git push origin main

run script *args:
    git fetch origin
    git reset --hard origin/main
    PYTHONPATH=. uv run python {{script}} {{args}}

[working-directory: 'notes']
build-update-slides:
    pandoc notes.md -t dzslides -s -H style.html --slide-level=2 -o notes.html

[working-directory: 'notes']
build-slides:
    pandoc phgn_pixels.md -t dzslides -s -H style.html --slide-level=2 --mathjax="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" -o phgn_pixels.html
