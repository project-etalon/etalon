.PHONY: help lint format
.DEFAULT_GOAL := help

lint/black: ## check style with black
	black --check etalon

lint/isort: ## check style with isort
	isort --check-only --profile black etalon

lint/autoflake: ## check for unused imports
	autoflake --recursive --remove-all-unused-imports --check etalon

lint/pyright: ## run type checking
	pyright

lint/codespell:
	codespell --skip './env/**,./docs/_build/**' -L inout

lint: lint/isort lint/black lint/autoflake lint/codespell lint/pyright	## check style

format/black: ## format code with black
	black etalon

format/isort: ## format code with isort
	isort --profile black etalon

format/autoflake: ## remove unused imports
	autoflake --in-place --recursive --remove-all-unused-imports etalon

format: format/isort format/autoflake format/black ## format code
