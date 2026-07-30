PYTHON ?= python
CONFIG ?= config/settings.yaml

.PHONY: install preprocess train evaluate infer app check test

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

preprocess:
	$(PYTHON) processing/preprocess.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

infer:
	$(PYTHON) scripts/infer.py --config $(CONFIG)

app:
	streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

check:
	$(PYTHON) scripts/check_project.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q
