.PHONY: install check test lint typecheck

install:
	python -m pip install -e '.[dev]'

check: lint typecheck test

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy
