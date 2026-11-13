#!/bin/sh
# Builds main.pdf (no notes) and main-notes.pdf (slides + speaker-notes pages). Same as `make all`.
set -e
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -jobname=main-notes -usepretex='\def\withnotes{}' main.tex
