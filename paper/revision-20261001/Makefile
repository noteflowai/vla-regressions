all: output/pdf/higher-success-new-failures.pdf arxiv.tar.gz reproducibility.tar.gz verification.json

figure-provenance.json: build_figures.py ../../analyze.py ../../results/pilot/*.jsonl ../../results/followup/*.jsonl ../../results/vqb/reanalysis.json
	python3 build_figures.py >figure-build-summary.txt

output/pdf/higher-success-new-failures.pdf: main.tex references.bib corl_2026.sty corlabbrvnat.bst figure-provenance.json figures/*.pdf
	mkdir -p .build output/pdf
	pdflatex -halt-on-error -interaction=nonstopmode -output-directory=.build main.tex >.build/pass1.txt
	cd .build && BIBINPUTS=.. BSTINPUTS=.. bibtex main >bibtex.txt
	pdflatex -halt-on-error -interaction=nonstopmode -output-directory=.build main.tex >.build/pass2.txt
	pdflatex -halt-on-error -interaction=nonstopmode -output-directory=.build main.tex >.build/pass3.txt
	cp .build/main.pdf output/pdf/higher-success-new-failures.pdf

arxiv.tar.gz: output/pdf/higher-success-new-failures.pdf prepare_sources.py
	python3 prepare_sources.py

reproducibility.tar.gz: prepare_reproducibility.py build_figures.py figure-provenance.json README.md ../../analyze.py ../../reanalyze_vqb.py ../../tests/test_paired_analysis.py ../../results/pilot/*.jsonl ../../results/followup/*.jsonl ../../results/vqb/reanalysis.*
	python3 prepare_reproducibility.py

verification.json: arxiv.tar.gz reproducibility.tar.gz verify_artifacts.py
	python3 verify_artifacts.py

.PHONY: all
