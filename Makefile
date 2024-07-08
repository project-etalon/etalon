.PHONY: help lint lint/black lint/isort format format/black format/isort
.DEFAULT_GOAL := help

lint/black: ## check style with black
	black --check metron

lint/isort: ## check style with isort
	isort --check-only --profile black metron

lint: lint/black lint/isort ## check style

format/black: ## format code with black
	black metron

format/isort: ## format code with isort
	isort --profile black metron

format: format/isort format/black ## format code
