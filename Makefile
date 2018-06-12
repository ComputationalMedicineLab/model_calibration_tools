MODULE_PATHS=`python -c "import os, sys; print(' '.join('{}'.format(d) for d in sys.path if os.path.isdir(d)))"`

all: clean install-reqs install-dev

install-dev:
	@echo "Installing in development mode"
	pip install -e .

install-reqs:
	@echo "Installing with pip:"
	@echo pip --version
	pip install --upgrade pip wheel setuptools
	pip install -r requirements.txt

clean:
	find . -type f -name \*.pyc -delete
	find . -type f -name \*.pyo -delete
	find . -type d -name __pycache__ -delete
	rm -rf .ipynb_checkpoints/

tags:
	ctags -R --fields=+l --extra=+f --languages=python --python-kinds=-iv -f ./tags $(MODULE_PATHS)
