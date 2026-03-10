install:
	uv sync
	pip install pydantic numpy
	pip install -e .

run:
	HF_HOME=/tmp/hf-home UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/call-me-maybe-venv uv run python -m src

debug:
	uv run python -m pdb src/__main__.py

clean:
	rm -f data/output/function_calling_results.json
	rm -f uv.lock
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

.PHONY: install run debug clean lint lint-strict