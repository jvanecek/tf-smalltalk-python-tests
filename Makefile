
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  make tests-for-smalltalk       - Run the test used to build the Smalltalk framework"
	@echo "  make tests-for-exploration     - Run the additional tests used to explore TensorFlow"
	@echo "  make thesis-experiments        - Run the thesis experiments
	@echo "  make install-requirements      - Install dependencies"
	@echo "  make clean                     - Clean up logs and cache directories"
	@echo "  make help                      - Show this help message"

.PHONY: tests-for-exploration
tests-for-exploration:
	pytest tests-for-exploration \
		--verbose \
		--junitxml=tests-for-exploration-result.xml

.PHONY: tests-for-smalltalk
tests-for-smalltalk:
	pytest tests-for-smalltalk \
		--verbose \
		--junitxml=tests-for-smalltalk-result.xml

.PHONY: thesis-experiments
thesis-experiments:
	python thesis-experiments/experiment1.py
	python thesis-experiments/experiment2.py
	python thesis-experiments/experiment3.py

.PHONY: plot-thesis-experiments
plot-thesis-experiments:
	python thesis-experiments/plot-metrics.py

.PHONY: install-requirements
install-requirements:
	pip install --upgrade pip
	pip install -r requirements.txt

.PHONY: clean
clean:
	@echo "Cleaning up directories..."
	find . -type d -name 'logs' -exec rm -r {} +
	find . -type d -name '.mypy_cache' -exec rm -r {} +
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -type d -name '.pytest_cache' -exec rm -r {} +
	@echo "Cleaning up files..."
	find . -type f -name '.coverage' -exec rm {} +
	find . -type f -name '*.py[co]' -delete
	find ./integration-tests -type f -name '*.xml' -exec rm {} +
	find ./tests -type f -name '*.xml' -exec rm {} +