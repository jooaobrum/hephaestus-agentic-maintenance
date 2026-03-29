run-docker-compose:
	uv sync
	docker compose up --build -d


run-evals:
	uv sync
	PYTHONPATH="$(CURDIR)/apps/api/src" uv run python apps/api/evals/eval.py

