verify:
	python code/check_reported_values.py

simulations:
	python code/simulation/run_all.py

figures:
	python code/figures/make_erie_figures.py
	python code/simulation/make_figures.py
