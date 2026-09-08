.PHONY: default help docs check-docs clean format lint lint-fix install

default: help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help                  Show this help message"
	@echo "  docs                  Build documentation (sphinx)"
	@echo "  clean                 Delete build artifacts"
	@echo "  format                Format code (ruff)"
	@echo "  lint                  Check code for linting issues (ruff)"
	@echo "  lint-fix              Fix issues found by linter (ruff)"
	@echo "  type-check            Check types (ty)"
	@echo "  check-docs            Check links in documentation"
	@echo "  install               Install dependencies"

docs:
	@echo "Building documentation..."
	@cd doc && make html

live-docs:
	@echo "Building & serving documentation live..."
	@cd doc && make livehtml

check-docs:
	@echo "Checking documentation..."
	@cd doc && make linkcheck

clean:
	@echo "Deleting build artifacts..."
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	@echo "Deleting release artifacts..."
	@rm -rf ./dist
	@echo "Deleting documentation build artifacts..."
	@cd doc && make clean

format:
	@uv run ruff format

lint:
	@uv run ruff check .

lint-fix:
	@uv run ruff check --fix .

type-check:
	@uv run ty check

install:
	@echo "Installing dependencies (incl. dev dependencies)..."
	@uv sync --all-extras --all-groups
