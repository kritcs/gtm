sim:
	uv run python -m sim.build

test:
	uv run pytest -q

run:
	uv run uvicorn api.main:app --port 8000

# static copy for GitHub Pages (writes dist/); make pages also pushes it to the gh-pages branch
static:
	uv run python scripts/build_static.py

pages:
	uv run python scripts/build_static.py --publish
