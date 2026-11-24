sim:
	uv run python -m sim.build

test:
	uv run pytest -q

run:
	uv run uvicorn api.main:app --port 8000
