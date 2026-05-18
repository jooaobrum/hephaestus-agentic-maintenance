run-docker-compose:
	uv sync
	docker compose up --build -d

run-pipeline:
	uv run python scripts/orchestrate_pipeline.py all

run-interventions:
	uv run python scripts/orchestrate_pipeline.py interventions

run-procedures:
	uv run python scripts/orchestrate_pipeline.py procedures

run-evals:
	PYTHONPATH="apps/api/src" uv run python apps/api/evals/eval.py

run-agent-evals:
	PYTHONPATH="apps/api/src" uv run python apps/api/evals/eval_agents.py
