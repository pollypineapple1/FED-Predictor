# Command launchers 

TESTS = TESTS

install:
	pip install --upgrade pip
	pip install -r requirements_local.txt

test:
	@pytest -v test