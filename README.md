# Centrelec Data Vault
Flask web app to upload Excel revisions, compare versions (added/deleted/modified),
view element history, and check budgets with a pricebook.

## Quickstart
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py   # http://127.0.0.1:5000/

## Env (optional)
# .env
DATABASE_URL=sqlite:///data.db
FLASK_ENV=development
