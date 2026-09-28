# main.pdf: anonymous submission. main-preprint.pdf: arXiv version with author names.
all: main.pdf main-preprint.pdf

main.pdf: main.tex references.bib ../fig_noise_floor.pdf
	pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main

main-preprint.pdf: main.tex references.bib ../fig_noise_floor.pdf
	pdflatex -interaction=nonstopmode -jobname=main-preprint "\def\PREPRINT{}\input{main}" && bibtex main-preprint && \
	pdflatex -interaction=nonstopmode -jobname=main-preprint "\def\PREPRINT{}\input{main}" && \
	pdflatex -interaction=nonstopmode -jobname=main-preprint "\def\PREPRINT{}\input{main}"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf

.PHONY: all clean
