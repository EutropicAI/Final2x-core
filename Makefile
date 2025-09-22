.DEFAULT_GOAL := default

.PHONY: test
test:
	uv run pytest --cov=Final2x_core --cov-report=xml --cov-report=html

.PHONY: lint
lint:
	uv run pre-commit install
	uv run pre-commit run --all-files

.PHONY: gen
gen:
	uv run python scripts/gen_config.py

.PHONY: gen-ts
gen-ts:
	uv run python scripts/gen_pretrained_model_name.py

.PHONY: build
build:
	uv build --wheel

.PHONY: pyinstaller
pyinstaller:
	uv run pyinstaller -n Final2x-core -i assets/favicon.ico Final2x_core/__main__.py
	uv run python scripts/post_pyinstaller.py
