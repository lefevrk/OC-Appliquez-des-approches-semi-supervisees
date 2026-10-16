#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = OC-Appliquez-des-approches-semi-supervisees
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (check only)
.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .

## Run all tests
.PHONY: test
test:
	uv run pytest

## Run all tests with coverage report (htmlcov/)
.PHONY: coverage
coverage:
	uv run pytest --cov=src --cov-report=html --cov-report=term-missing

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Git Hooks
.PHONY: setup_hooks
setup_hooks:
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push


## Make dataset
.PHONY: data
data: requirements
	uv run src/brain_mri/dataset.py


## Extract features
.PHONY: features
features: requirements
	uv run src/brain_mri/features.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
