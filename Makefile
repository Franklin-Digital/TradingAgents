# TauricTradingAgents — build, test, deploy
#
# Usage:
#   make test            Run unit tests
#   make test-all        Run all tests (unit + integration + smoke)
#   make install         pip install in editable mode
#   make docker-build    Build the Docker image
#   make docker-up       Start containers (tradingagents + optional ollama)
#   make docker-down     Stop containers
#   make deploy          Rsync repo to mac-pro production path

.PHONY: test test-all install lint docker-build docker-up docker-down deploy

PYTHON     := python3
PROD       := /home/Franklin/TauricTradingAgents

# ── Test ──────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v -m unit

test-all:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m py_compile tradingagents/cli/main.py
	$(PYTHON) -m py_compile main.py

# ── Install ───────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install -e .

# ── Docker ────────────────────────────────────────────────────────────

docker-build:
	docker compose build tradingagents

docker-up:
	docker compose up -d tradingagents

docker-down:
	docker compose down

# ── Deploy (run ON mac-pro) ───────────────────────────────────────────

deploy:
	@echo "=== Pulling latest TauricTradingAgents ==="
	cd $(PROD) && git pull
	@echo "=== Done ==="
