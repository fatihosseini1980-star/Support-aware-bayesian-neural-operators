PYTHON ?= python

.PHONY: audit check figures manuscript

audit:
	$(PYTHON) code/erie/lake_erie_audit.py

check:
	$(PYTHON) code/check_reported_values.py

figures:
	$(PYTHON) code/figures/make_erie_figures.py
	$(PYTHON) code/figures/plot_simulation_summary.py

manuscript:
	cd manuscript && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
	cd manuscript && pdflatex -interaction=nonstopmode supplement.tex && pdflatex -interaction=nonstopmode supplement.tex
