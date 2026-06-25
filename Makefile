# Makefile — insurance-funnel-analytics
# Usage: make <target>

PYTHON = python

.PHONY: all install simulate notebooks dashboard clean test

## Run full pipeline (install → simulate → notebooks → dashboard)
all: install simulate notebooks dashboard

## Install Python dependencies
install:
	pip install -r requirements-dev.txt

## Install dashboard-only dependencies for Streamlit deployment
install-app:
	pip install -r requirements.txt

## Generate synthetic funnel dataset
simulate:
	$(PYTHON) data/simulate_funnel.py

## Run all analysis notebooks in order
notebooks:
	mkdir -p app/assets models
	$(PYTHON) notebooks/01_eda.py
	$(PYTHON) notebooks/02_survival_analysis.py
	$(PYTHON) notebooks/03_abandonment_model.py
	$(PYTHON) notebooks/04_intervention_sim.py

## Launch Streamlit dashboard
dashboard:
	$(PYTHON) -m streamlit run app/dashboard.py

## Convert .py notebooks to .ipynb
ipynb:
	$(PYTHON) convert_notebooks.py

## Run pipeline without launching dashboard
ci: install simulate notebooks

## Run tests
test:
	@if [ -d tests ]; then \
		$(PYTHON) -m pytest tests/ -v; \
	else \
		echo "No tests/ directory found; skipping pytest."; \
	fi

## Remove generated artefacts
clean:
	rm -f data/funnel_events.csv
	rm -f data/scored_sessions.csv
	rm -f models/abandonment_xgb.pkl
	rm -f app/assets/*.png
	rm -f notebooks/*.ipynb
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

## Show available targets
help:
	@echo ""
	@echo "  make install    — install dependencies"
	@echo "  make simulate   — generate funnel_events.csv"
	@echo "  make notebooks  — run all 4 analysis notebooks"
	@echo "  make dashboard  — launch Streamlit at localhost:8501"
	@echo "  make ipynb      — convert .py notebooks to .ipynb"
	@echo "  make ci         — simulate + notebooks (no dashboard)"
	@echo "  make test       — run pytest suite"
	@echo "  make clean      — remove all generated files"
	@echo ""