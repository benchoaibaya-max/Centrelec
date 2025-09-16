# API (high level)

> Routes are implemented in pp.py and rendered with Jinja2 templates.

## Upload
- **POST /upload-many**
  - Body: multipart form iles[] (.xlsx), optional key
  - Result: JSON { created: [{id, name}], error? }

## Revisions
- **GET /revisions**: list all revisions (HTML)
- **POST /revisions/<id>/delete**: remove a revision (+ its snapshots) → JSON {ok:true}

## Compare
- **GET /compare**: compare UI
- **POST /compare**: JSON payload with baseline + targets → returns diff JSON used by the page
- **POST /export_xlsx**: downloads an .xlsx report of the current diff

## Elements & History
- **GET /elements**: list unique keys and their revision presence
- **GET /elements/<key>**: per-key snapshots and between-revision changes

## Budget
- **GET /budget**: budget UI
- **POST /pricebook/upload**: upload/update pricebook (.xlsx)
- **GET /api/revision/<id>/budget_by_designation**: JSON of computed totals

## Notes
- DB is SQLite by default (data.db). For Postgres, set DATABASE_URL.
- Excel parsing uses Pandas/openpyxl; export uses openpyxl.
