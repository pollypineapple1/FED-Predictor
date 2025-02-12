# Command launchers 

TESTS = TESTS

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	@pytest -v test