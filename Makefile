PACKAGE=mct
MODULE_PATHS=`python -c "import os, sys; print(' '.join('{}'.format(d) for d in sys.path if os.path.isdir(d)))"`

.PHONY: all clean install test tags

all: clean install test
clean: uninstall-dev clean-pyc clean-build clean-artifacts
install: install-reqs install-dev

install-dev:
	@echo "Installing in development mode"
	pip install -e .

install-reqs:
	@echo "Installing with pip:"
	@echo pip --version
	pip install --upgrade pip wheel setuptools
	pip install -r requirements.txt

uninstall-dev:
	yes | pip uninstall $(PACKAGE)

clean-pyc:
	find . -type f -name \*.pyc -delete
	find . -type f -name \*.pyo -delete
	find . -type d -name __pycache__ -delete

clean-build:
	rm -rf *.egg-info/

clean-artifacts:
	rm -rf .ipynb_checkpoints/
	rm -rf .pytest_cache/
	rm -rf .coverage/
	rm -rf .cache/
	rm -rf htmlcov/
	rm -f tags

smart-freeze: uninstall-dev
	pip freeze > requirements.txt
	make install-dev --no-print-directory

test: install-dev
	pytest -vv --cov=mct/

htmlcov: install-dev
	pytest -vv --cov=mct/ --cov-report=html
	cd htmlcov && python -m http.server

tags:
	ctags -R --fields=+l --extra=+f --languages=python --python-kinds=-iv -f ./tags $(MODULE_PATHS)
