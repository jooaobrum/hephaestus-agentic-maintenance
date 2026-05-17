run-docker-compose:
	uv sync
	docker compose up --build -d

run-pipeline:
	uv run python scripts/orchestrate_pipeline.py all

run-interventions:
	uv run python scripts/orchestrate_pipeline.py interventions

run-procedures:
	uv run python scripts/orchestrate_pipeline.py procedures
