# Requires a TeX Live with beamer + metropolis + Fira fonts (built with xelatex).
LATEXMK = latexmk -xelatex -interaction=nonstopmode -halt-on-error

all: main.pdf main-notes.pdf

main.pdf: main.tex figures/*.pdf
	$(LATEXMK) main.tex

main-notes.pdf: main.tex figures/*.pdf
	$(LATEXMK) -jobname=main-notes -usepretex='\def\withnotes{}' main.tex

notes: main-notes.pdf


clean:
	latexmk -c main.tex; latexmk -c -jobname=main-notes main.tex

.PHONY: all notes clean
