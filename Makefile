
install:
	python3 -m venv .venv
	. .venv/bin/activate && uv sync && pip install pydantic numpy && pip install mypy flake8 && pip install -e .

run:
	HF_HOME=/tmp/hf-home UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run --active python -m src

debug:
	uv run python -m pdb src/__main__.py

clean:
	rm -f data/output/function_calling_results.json
	rm -f uv.lock
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type d -name ".venv" -prune -exec rm -rf {} +
lint:
	python3 -m flake8
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	python3 -m flake8
	mypy . --strict

.PHONY: install run debug clean lint lint-strict