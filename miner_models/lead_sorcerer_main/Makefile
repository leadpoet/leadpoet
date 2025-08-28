.PHONY: help install install-dev test test-cov lint format clean gc

help: ## Show this help message
	@echo "Lead Sorcerer - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	poetry install --no-dev

install-dev: ## Install development dependencies
	poetry install

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting and type checking
	poetry run pre-commit run --all-files
	poetry run mypy src/

format: ## Format code
	poetry run black src/ tests/
	poetry run isort src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

gc: ## Run garbage collection
	poetry run python scripts/garbage_collector.py

install-hooks: ## Install pre-commit hooks
	poetry run pre-commit install

update-deps: ## Update dependencies
	poetry update

build: ## Build the package
	poetry build

publish: ## Publish to PyPI (use with caution)
	poetry publish

check: ## Run all checks (lint, test, type check)
	$(MAKE) lint
	$(MAKE) test

ci: ## Run CI checks locally
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-cov
