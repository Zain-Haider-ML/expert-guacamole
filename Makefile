install:
	pip install -r requirements.txt
lint:
	pylint --disable=R,C,no-member --max-line-length=120 train.py test.py
format:
	black --line-length 120 train.py test.py
	isort train.py test.py
test:
	python -m pytest -vv --cov=train test.py