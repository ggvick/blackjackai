.PHONY: test lint format

test:
	pytest -q

lint:
	ruff check blackjackai_rl tests
	black --check blackjackai_rl tests

format:
	black blackjackai_rl tests
