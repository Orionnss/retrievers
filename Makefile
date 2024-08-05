clean:
	rm -rf dist build *.egg-info *__pycache__ .pytest_cache

build:
	pip-compile pyproject.toml
	python -m build

install:
	pip install .