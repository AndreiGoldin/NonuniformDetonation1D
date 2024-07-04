typehint:
	mypy src/*.py tests/*.py

test:
	pytest tests/*.py

lint:
	pylint src/*.py tests/*.py

checklist: lint typehint test

black:
	black -l 80 src/*.py

clean:
	rm -r *pics *data *videos

.PHONY: typehint test lint checklist black clean

install:
    python3 -m venv solver-env
    source solver-env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt