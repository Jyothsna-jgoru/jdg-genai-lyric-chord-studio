PYTHON ?= python

.PHONY: install dataset-dev dataset-full validate-data train-dev train-full evaluate test serve docker
install:
	$(PYTHON) -m pip install -e ".[dev]"
dataset-dev:
	$(PYTHON) -m app.cli dataset-generate --config configs/dataset_dev.toml
dataset-full:
	$(PYTHON) -m app.cli dataset-generate --config configs/dataset_full.toml
validate-data:
	$(PYTHON) -m app.cli dataset-validate --dataset-dir data/generated/dev
train-dev:
	$(PYTHON) -m app.cli train --config configs/train_dev.toml
train-full:
	$(PYTHON) -m app.cli train --config configs/train_full.toml
evaluate:
	$(PYTHON) -m app.cli evaluate --config configs/train_dev.toml
test:
	$(PYTHON) -m pytest
serve:
	$(PYTHON) -m app.cli serve
docker:
	docker compose up --build

