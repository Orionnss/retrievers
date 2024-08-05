clean:
	rm -rf dist build *.egg-info *__pycache__ .pytest_cache

build:
	pip-compile pyproject.toml --extra=dev
	python -m build

run-tests:
	python -m pytest src/tests

install:
	pip install .