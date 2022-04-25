typehint:
	mypy src/*.py tests/*.py

test:
	pytest tests/*.py

lint:
	pylint src/*.py tests/*.py

checklist: lint typehint test

black:
	black -l 80 src/*.py tests/*.py

.PHONY: typehint test lint checklist black
