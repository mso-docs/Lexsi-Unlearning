# Requires a TeX Live with beamer + metropolis + Fira fonts (built with xelatex).
LATEXMK = latexmk -xelatex -interaction=nonstopmode -halt-on-error

all: main.pdf main-notes.pdf

main.pdf: main.tex figures/*.png
	$(LATEXMK) main.tex

main-notes.pdf: main.tex figures/*.png
	$(LATEXMK) -jobname=main-notes -usepretex='\def\withnotes{}' main.tex

notes: main-notes.pdf

# Re-crop the synthetic dashboard screenshots (needs Python + Pillow; see figures/make_figs.py)
figs:
	python3 figures/make_figs.py

clean:
	latexmk -c main.tex; latexmk -c -jobname=main-notes main.tex

.PHONY: all notes figs clean
