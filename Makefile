.PHONY: test lint format install

install:
pip install -e ./blackjack_env

lint:
ruff check blackjack_env agents tests
black --check blackjack_env agents tests

format:
black blackjack_env agents tests

test:
pytest -q
