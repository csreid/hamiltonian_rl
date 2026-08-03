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
    pandoc notes.md -t revealjs -s -c reveal-style.css --slide-level=2 \
        --mathjax="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" \
        -V revealjs-url=https://cdn.jsdelivr.net/npm/reveal.js@5 \
        -V theme=moon \
        -o notes.html

[working-directory: 'notes']
build-arch-slides:
    pdflatex -interaction=nonstopmode -halt-on-error phamiltonian_arch.tex
    pdflatex -interaction=nonstopmode -halt-on-error slides.tex
    rm -f phamiltonian_arch.aux phamiltonian_arch.log \
        slides.aux slides.log slides.nav slides.out slides.snm slides.toc slides.vrb

[working-directory: 'notes']
build-loss-pdf:
    pdflatex -interaction=nonstopmode -halt-on-error loss_functions.tex
    rm -f loss_functions.aux loss_functions.log

[working-directory: 'notes']
build-slides:
    pandoc phgn_pixels.md -t dzslides -s -H style.html --slide-level=2 --mathjax="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" -o phgn_pixels.html
