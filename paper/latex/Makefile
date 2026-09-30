# main.pdf: anonymous submission. main-preprint.pdf: arXiv version with author names.
all: main.pdf main-preprint.pdf

main.pdf: main.tex references.bib ../fig_noise_floor.pdf
	pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main

main-preprint.pdf: main.tex references.bib ../fig_noise_floor.pdf
	pdflatex -interaction=nonstopmode -jobname=main-preprint "\def\PREPRINT{}\input{main}" && bibtex main-preprint && \
	pdflatex -interaction=nonstopmode -jobname=main-preprint "\def\PREPRINT{}\input{main}" && \
	pdflatex -interaction=nonstopmode -jobname=main-preprint "\def\PREPRINT{}\input{main}"

# arXiv source: the preprint build as one flat directory. arXiv does not run BibTeX, so the .bbl
# goes along. Check the tarball's content before uploading it.
arxiv.tar.gz: main-preprint.pdf prepare_arxiv_sources.py
	rm -rf arxiv && mkdir arxiv
	python3 prepare_arxiv_sources.py --output-dir arxiv
	cd arxiv && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null

clean:
	rm -rf *.aux *.bbl *.blg *.log *.out *.pdf arxiv arxiv.tar.gz

.PHONY: all clean
