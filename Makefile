.PHONY: help lint lint/black lint/isort format format/black format/isort
.DEFAULT_GOAL := help

lint/black: ## check style with black
	black --check etalon

lint/isort: ## check style with isort
	isort --check-only --profile black etalon

lint: lint/black lint/isort ## check style

format/black: ## format code with black
	black etalon

format/isort: ## format code with isort
	isort --profile black etalon

format: format/isort format/black ## format code
