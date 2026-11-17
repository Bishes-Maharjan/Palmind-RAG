.PHONY: help up down test lint typecheck format run migrate

SHELL := /bin/bash
VENV := source ~/myenv/bin/activate

help: ## Show help messages
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

up: ## Start all infrastructure services and API with Docker Compose
	docker-compose up --build -d

down: ## Stop and remove all containers, networks, and volumes
	docker-compose down

run: ## Run API locally with uvicorn (requires local or compose dependencies)
	$(VENV) && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

migrate: ## Run Alembic database migrations
	$(VENV) && alembic upgrade head

test: ## Run test suite with pytest
	$(VENV) && pytest -v

lint: ## Run code linter checks (ruff)
	$(VENV) && ruff check .

format: ## Format code using ruff
	$(VENV) && ruff format . && ruff check --fix .

typecheck: ## Run static type checking with mypy
	$(VENV) && mypy app tests
