sync:
	uv sync --all-groups
sample:
	uv run python scripts/create_sample_data.py
validate:
	uv run revenue-forecast validate
run:
	uv run revenue-forecast run-all --year 2027
test:
	uv run pytest -q
lint:
	uv run ruff check .
