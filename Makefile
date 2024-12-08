VENV = venv
FLASK_APP = app.py
REQUIREMENTS = requirements.txt

install:
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r $(REQUIREMENTS)

run: install
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port=3000

clean:
	rm -rf $(VENV)

reinstall: clean install
