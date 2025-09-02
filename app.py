#app.py
import os, json, unicodedata, re, datetime, io
from typing import Dict, Optional, List
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# ---------- Flask / DB ----------
DB_URL = os.getenv("DATABASE_URL", "sqlite:///data.db")
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URL
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-prod")

UPLOADS = os.getenv("UPLOADS_DIR", "uploads")
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads
db = SQLAlchemy(app)
# after db = SQLAlchemy(app)
def _init_schema_once():
    # guard so the dev reloader doesn't run this twice
    if getattr(app, "_SCHEMA_INIT_DONE", False):
        return
    with app.app_context():
        db.create_all()
        ensure_schema()
    app._SCHEMA_INIT_DONE = True


UPLOADS = "uploads"
os.makedirs(UPLOADS, exist_ok=True)


# ---------- Models ----------
class Revision(db.Model):
    __tablename__ = "revisions"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    total_amount = db.Column(db.Float, nullable=True)    # cached sum of the last budget check
    over_budget  = db.Column(db.Boolean, default=False)  # cached flag

class Element(db.Model):
    __tablename__ = "elements"
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(255), unique=True, index=True, nullable=False)  # stable key (e.g., N° de Prix)
    first_seen_rev_id = db.Column(db.Integer, db.ForeignKey("revisions.id"), nullable=True)

class Snapshot(db.Model):
    __tablename__ = "snapshots"
    id = db.Column(db.Integer, primary_key=True)
    revision_id = db.Column(db.Integer, db.ForeignKey("revisions.id"), index=True, nullable=False)
    element_id = db.Column(db.Integer, db.ForeignKey("elements.id"), index=True, nullable=False)
    data_json = db.Column(db.Text, nullable=False)  # stores all columns for that row as JSON
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class PriceBook(db.Model):
    __tablename__ = "pricebooks"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class PriceItem(db.Model):
    __tablename__ = "priceitems"
    id = db.Column(db.Integer, primary_key=True)
    pricebook_id = db.Column(db.Integer, db.ForeignKey("pricebooks.id"), index=True, nullable=False)
    designation = db.Column(db.String(1024), index=True, nullable=False)   # raw text from the file
    unit = db.Column(db.String(64), nullable=True)                         # e.g., 'ML', 'U', 'E'
    unit_price = db.Column(db.Float, nullable=True)

# ---------- Utilities (normalize, key detection, diff) ----------
ID_ALIASES = [
    "n° de prix","n de prix","no de prix","numero de prix","num de prix",
    "id","code","code article","reference","réference","ref","ref article",
    "designation","désignation","libelle","libellé","name","nom"
]

def norm_text(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+"," ", s.strip().lower())
    return s

def norm_col(c: str) -> str:
    s = norm_text(c).replace("°","").replace("º","")
    s = s.replace("n de", "no de").replace("n ", "no ")
    return s

def make_unique(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0; out.append(c)
        else:
            seen[c] += 1; out.append(f"{c}__{seen[c]}")
    return out

def find_key_column(columns: List[str]) -> Optional[str]:
    """Return the normalized name of the best key column, if any."""
    normed = [norm_col(c) for c in columns]
    # 1) exact alias match
    for alias in ID_ALIASES:
        if alias in normed: return alias
    # 2) heuristic
    for i, n in enumerate(normed):
        if any(k in n for k in ["prix","id","code","ref","design","libelle"]):
            return n
    return None

def coerce_num(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if s == "": return None
    s = s.replace("\xa0"," ").replace(" ","")
    if "," in s and "." not in s: s = s.replace(",",".")
    try: return float(s)
    except: return None

def values_differ(a,b, tol=1e-9):
    if pd.isna(a) and pd.isna(b): return False
    fa, fb = coerce_num(a), coerce_num(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) > tol
    sa = "" if pd.isna(a) else str(a).strip()
    sb = "" if pd.isna(b) else str(b).strip()
    return sa != sb

def diff_rows(old: Dict, new: Dict) -> List[Dict]:
    """Return list of changes [{'field','old','new'}] between two dicts."""
    changes = []
    keys = sorted(set(old.keys()) | set(new.keys()))
    for k in keys:
        if k == "_key": continue
        a = old.get(k); b = new.get(k)
        if values_differ(a,b):
            changes.append({"field": k, "old": a, "new": b})
    return changes

DESIGN_ALIASES = ["designation","désignation","libelle","libellé","name","nom"]
UNIT_ALIASES = ["u","unite","unité","unit","unity"]
QTY_ALIASES_GENERIC = ["quantite","quantité","qte","qté","qty","quant","quantites","quantités"]
QTY_ALIASES_RECEP   = ["nbr de recepteur","nbre de recepteur","nb recepteur","nombre de recepteur",
                       "nbr recepteur","nbre recepteurs","nb recepteurs","nombre de recepteurs"]
LENGTH_ALIASES      = ["longueur","long","length","len","m","ml","metres","mètres","m\u00e8tres"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {norm_col(c): c for c in df.columns}
    for alias in candidates:
        if alias in cols:
            return cols[alias]
    for nc, orig in cols.items():
        if any(alias in nc for alias in candidates):
            return orig
    return None

def canon(s: str) -> str:
    """Accent-insensitive, lowercase, single-space, strips some punctuation."""
    x = norm_text(s)
    x = re.sub(r"[()\[\]{}]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _to_float(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if not s: return None
    s = re.sub(r"[^\d,.-]", "", s)
    if "," in s and s.count(",")==1 and "." not in s:
        s = s.replace(",", ".")
    try: return float(s)
    except: return None

def ingest_pricebook(file_storage, forced_designation_col: Optional[str]=None) -> PriceBook:
    filename = file_storage.filename or "pricebook.xlsx"
    save_path = os.path.join(UPLOADS, filename)
    file_storage.save(save_path)

    df = pd.read_excel(save_path, engine="openpyxl")
    df = df.rename(columns={c: norm_col(c) for c in df.columns})
    df.columns = make_unique(list(df.columns))

    col_design = norm_col(forced_designation_col) if forced_designation_col else _find_col(df, DESIGN_ALIASES)
    if not col_design:
        raise ValueError("Bordereau: colonne 'Désignation' introuvable.")

    col_unit_price = _find_col(df, PRICE_ALIASES_UNIT)
    if not col_unit_price:
        raise ValueError("Bordereau: colonne 'Prix unitaire' introuvable (e.g. 'Prix unitaire', 'PU').")

    col_unit = _find_col(df, UNIT_ALIASES)

    pb = PriceBook(name=os.path.splitext(os.path.basename(filename))[0])
    db.session.add(pb); db.session.flush()

    for _, row in df.iterrows():
        desig = str(row.get(col_design, "")).strip()
        if not desig:
            continue
        db.session.add(PriceItem(
            pricebook_id = pb.id,
            designation  = desig,
            unit         = (None if not col_unit else (None if pd.isna(row.get(col_unit)) else str(row.get(col_unit)))),
            unit_price   = _to_float(row.get(col_unit_price))
        ))
    db.session.commit()
    return pb


def _find_qty_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, QTY_ALIASES)

def _pick_qty_column_for_revision(df_rev: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Find best columns for qty: prefer Longueur for cables, then Nbr de récepteur, else generic Quantité."""
    col_len   = _find_col(df_rev, LENGTH_ALIASES)
    col_recv  = _find_col(df_rev, QTY_ALIASES_RECEP)
    col_qty   = _find_col(df_rev, QTY_ALIASES_GENERIC)
    col_desig = _find_col(df_rev, DESIGN_ALIASES)
    return {"len": col_len, "recv": col_recv, "qty": col_qty, "desig": col_desig}

def _is_cable_row(designation: str, unit_hint: Optional[str]) -> bool:
    d = canon(designation)
    if "cable" in d or "câble" in designation.lower():
        return True
    if unit_hint:
        u = canon(unit_hint)
        # most pricebooks use ML/m/… for length-based entries
        if u in ("ml","m","metres","metre","mètres","m\u00e8tres"):
            return True
    return False

def compute_revision_budget_by_designation(rev_id: int, budget_limit: Optional[float]=None):
    """Use the latest PriceBook. Match on Désignation; choose qty rule per row."""
    # 1) Get latest pricebook
    pb = PriceBook.query.order_by(PriceBook.uploaded_at.desc()).first()
    if not pb:
        return {"error": "Aucun bordereau de prix trouvé. Veuillez l'uploader une seule fois."}

    # 2) Build price lookup (canon(designation) -> (price, unit))
    items = PriceItem.query.filter_by(pricebook_id=pb.id).all()
    price_map = {}
    for it in items:
        key = canon(it.designation)
        if key and key not in price_map:  # keep first
            price_map[key] = (it.unit_price, it.unit)

    # 3) Read revision rows
    df = df_from_revision(rev_id)
    if df is None or df.empty:
        return {"ok": True, "revision_id": rev_id, "pricebook_id": pb.id, "total": 0.0,
                "limit": budget_limit, "over_budget": False, "matched": 0, "missing": []}

    cols = _pick_qty_column_for_revision(df)
    col_desig = cols["desig"]
    if not col_desig:
        return {"error": "Révision: colonne 'Désignation' introuvable."}

    # 4) Iterate rows, choose qty
    missing, matched = [], 0
    totals = []

    for _, row in df.iterrows():
        desig_raw = str(row.get(col_desig, "")).strip()
        if not desig_raw:
            continue
        key = canon(desig_raw)
        pu, unit_hint = price_map.get(key, (None, None))

        if pu is None:
            missing.append(desig_raw)
            totals.append(0.0)
            continue

        # choose qty
        qty = None
        if _is_cable_row(desig_raw, unit_hint) and cols["len"]:
            qty = _to_float(row.get(cols["len"]))
        if qty is None:
            if cols["recv"]:
                qty = _to_float(row.get(cols["recv"]))
            if qty is None and cols["qty"]:
                qty = _to_float(row.get(cols["qty"]))
        if qty is None:
            qty = 1.0  # last resort

        matched += 1
        totals.append((pu or 0.0) * (qty or 0.0))

    total = float(pd.Series(totals).sum())
    over  = (budget_limit is not None) and (total > float(budget_limit))

    # cache on revision (optional)
    rev = Revision.query.get(rev_id)
    if rev:
        rev.total_amount = total
        rev.over_budget  = bool(over)
        db.session.commit()

    return {
        "ok": True,
        "revision_id": rev_id,
        "pricebook_id": pb.id,
        "total": round(total, 2),
        "limit": (round(float(budget_limit), 2) if budget_limit is not None else None),
        "over_budget": over if budget_limit is not None else None,
        "matched": matched,
        "missing": sorted(set(missing), key=canon)
    }


def compute_revision_budget_with_pricebook(rev_id: int, pricebook_id: int, budget_limit: Optional[float]=None):
    """Return verdict + breakdown using the specified pricebook."""
    # 1) get revision rows
    df_rev = df_from_revision(rev_id)
    if df_rev is None or df_rev.empty:
        return {
            "ok": True, "revision_id": rev_id, "pricebook_id": pricebook_id,
            "total": 0.0, "limit": budget_limit, "over_budget": False,
            "matched": 0, "missing_prices": []
        }

    # 2) build price map from pricebook
    items = PriceItem.query.filter_by(pricebook_id=pricebook_id).all()
    price_map = {canon(it.designation): (it.unit_price if it.unit_price is not None else None) for it in items}


    # 3) detect qty column in the revision
    qty_col = _find_qty_col(df_rev)
    # default qty = 1 if none
    def get_qty(row):
        if qty_col and qty_col in row:
            q = _to_float(row.get(qty_col))
            return q if (q is not None and q >= 0) else 0.0
        return 1.0

    # 4) compute line totals
    missing = []
    matched = 0
    line_totals = []
    for _, row in df_rev.iterrows():
        key = str(row.get("N° de Prix", "")).strip()
        if not key:
            continue
        pu = price_map.get(key)
        qty = get_qty(row)
        if pu is None:
            missing.append(key)
            line_totals.append(0.0)
        else:
            matched += 1
            line_totals.append((pu or 0.0) * (qty or 0.0))

    total = float(pd.Series(line_totals).sum())
    over = (budget_limit is not None) and (total > float(budget_limit))

    # cache on revision (optional)
    rev = Revision.query.get(rev_id)
    if rev:
        rev.total_amount = total
        rev.over_budget = bool(over)
        db.session.commit()

    return {
        "ok": True,
        "revision_id": rev_id,
        "pricebook_id": pricebook_id,
        "total": round(total, 2),
        "limit": (round(float(budget_limit),2) if budget_limit is not None else None),
        "over_budget": over if budget_limit is not None else None,
        "matched": matched,
        "missing_prices": sorted(set(missing), key=lambda s: s)
    }



# ---------- Ingestion ----------
def ingest_excel_to_revision(file_storage, forced_key: Optional[str]=None) -> Revision:
    filename = file_storage.filename or "upload.xlsx"
    save_path = os.path.join(UPLOADS, filename)
    file_storage.save(save_path)

    df = pd.read_excel(save_path, engine="openpyxl")
    # Normalize headers
    norm_map = {c: norm_col(c) for c in df.columns}
    df = df.rename(columns=norm_map)
    df.columns = make_unique(list(df.columns))

    # Detect key col (normalized)
    key_norm = norm_col(forced_key) if forced_key else find_key_column(list(df.columns))
    if not key_norm or key_norm not in df.columns:
        raise ValueError("Could not detect a stable key column (e.g., 'N° de Prix', 'Code', 'Ref', 'Designation').")

    # Create revision
    rev = Revision(name=os.path.splitext(os.path.basename(filename))[0])
    db.session.add(rev); db.session.flush()  # get rev.id

    # Ingest rows
    for _, row in df.iterrows():
        key_val = str(row.get(key_norm, "")).strip()
        if not key_val:  # skip empty keys
            continue

        # upsert element by key
        el = Element.query.filter_by(key=key_val).first()
        if not el:
            el = Element(key=key_val, first_seen_rev_id=rev.id)
            db.session.add(el); db.session.flush()

        # Build data dict (keep all normalized columns, include _key for convenience)
        data = {c: (None if pd.isna(row[c]) else row[c]) for c in df.columns}
        data["_key"] = key_val

        snap = Snapshot(revision_id=rev.id, element_id=el.id, data_json=json.dumps(data, ensure_ascii=False))
        db.session.add(snap)

    db.session.commit()
    return rev
# ---- Build a DataFrame from a stored revision (snapshots → rows)
def df_from_revision(rev_id: int) -> pd.DataFrame:
    snaps = Snapshot.query.filter_by(revision_id=rev_id).all()
    rows = [json.loads(s.data_json) for s in snaps]
    if not rows:
        return pd.DataFrame(columns=["N° de Prix"])
    df = pd.DataFrame(rows)
    # display ID from the stable key we stored during ingest
    if "_key" in df.columns:
        df["N° de Prix"] = df["_key"].astype(str).str.strip()
        df = df[df["N° de Prix"] != ""]
    if "N° de Prix" not in df.columns:
        df["N° de Prix"] = ""
    return df


# ---- Compare two DataFrames already containing "N° de Prix"
def analyze_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    ID_DISPLAY = "N° de Prix"
    if ID_DISPLAY not in df1.columns or ID_DISPLAY not in df2.columns:
        raise ValueError("Both dataframes must contain 'N° de Prix' column.")

    # Align columns
    all_cols = sorted(set(df1.columns) | set(df2.columns))
    for c in ("_key",):  # hidden technical cols
        if c in all_cols:
            all_cols.remove(c)
    ordered = [ID_DISPLAY] + [c for c in all_cols if c != ID_DISPLAY]

    A = df1.reindex(columns=ordered, fill_value=pd.NA).copy()
    B = df2.reindex(columns=ordered, fill_value=pd.NA).copy()

    sA = A[ID_DISPLAY].astype(str)
    sB = B[ID_DISPLAY].astype(str)

    # Added / Deleted
    added   = B[~sB.isin(sA)].copy().sort_values(by=ID_DISPLAY)
    deleted = A[~sA.isin(sB)].copy().sort_values(by=ID_DISPLAY)

    # Modified
    common_ids   = sorted(set(sA) & set(sB))
    A_common     = A.set_index(ID_DISPLAY).loc[common_ids]
    B_common     = B.set_index(ID_DISPLAY).loc[common_ids]
    cols_to_check = [c for c in A_common.columns if c in B_common.columns and c != ID_DISPLAY]

    mods = []
    for the_id in common_ids:
        row_old = A_common.loc[the_id]
        row_new = B_common.loc[the_id]
        for col in cols_to_check:
            a = row_old.get(col, pd.NA)
            b = row_new.get(col, pd.NA)
            if values_differ(a, b):
                mods.append({
                    ID_DISPLAY: the_id,
                    "Champ modifié": col,
                    "Ancienne valeur": None if pd.isna(a) else a,
                    "Nouvelle valeur": None if pd.isna(b) else b,
                })

    modified = (pd.DataFrame(mods, columns=[ID_DISPLAY, "Champ modifié", "Ancienne valeur", "Nouvelle valeur"])
                .sort_values([ID_DISPLAY, "Champ modifié"]) if mods else
                pd.DataFrame(columns=[ID_DISPLAY, "Champ modifié", "Ancienne valeur", "Nouvelle valeur"]))

    return added, deleted, modified

def df_to_records(df: pd.DataFrame):
    """Return JSON-safe list[dict] from a DataFrame (handles NaN, numpy types, dates)."""
    if df is None or df.empty:
        return []
    # Replace NaN with None, convert to proper JSON (pandas handles numpy types), then back to Python
    return json.loads(df.where(pd.notnull(df), None).to_json(orient="records", date_format="iso"))
def sort_records_by_id(records, id_key="N° de Prix"):
    def parse(val):
        s = "" if val is None else str(val).replace(",", ".").strip()
        try:
            return (0, float(s))  # numeric first
        except:
            return (1, s)         # then lexicographic for non-numerics
    return sorted(records, key=lambda r: parse(r.get(id_key)))

def clean_added_deleted_records(records):
    """Remove duplicate id columns like 'no de prix' and keep 'N° de Prix'."""
    cleaned = []
    for r in records:
        new = dict(r)
        for k in list(new.keys()):
            if k == "N° de Prix":
                continue
            # normalize similar to norm_col to catch 'no de prix'
            nk = norm_col(k)
            if nk == "no de prix":
                new.pop(k, None)
        cleaned.append(new)
    return cleaned
def _parse_num_for_sort(x):
    s = "" if x is None else str(x).replace(",", ".").strip()
    try:
        return (0, float(s))  # numeric first
    except:
        return (1, s)

def _drop_dup_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 'N° de Prix' and drop any column that normalizes to 'no de prix'."""
    if df is None or df.empty:
        return df
    keep = ["N° de Prix"]
    keep += [c for c in df.columns
             if c != "N° de Prix" and norm_col(c) != "no de prix"]
    return df.reindex(columns=[c for c in keep if c in df.columns])

def _sort_by_num_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "N° de Prix" not in df.columns:
        return df
    return df.sort_values(
        by="N° de Prix",
        key=lambda s: s.map(lambda v: _parse_num_for_sort(v))
    )
PRICE_ALIASES_TOTAL = [
    "montant", "total", "prix total", "total ttc", "total ht", "montant ttc", "montant ht",
    "prix_total", "amount", "line total"
]
PRICE_ALIASES_UNIT = [
    "prix unitaire", "pu", "p.u.", "unit price", "prix", "prix_u", "prixunitaire", "p u"
]
QTY_ALIASES = [
    "quantite", "quantité", "qte", "qté", "qty", "quant", "quantites", "quantités"
]

def compute_line_totals(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Returns a copy of df with a 'Montant_calculé' column + total sum.
    It uses 'Montant'/'Total' if present; otherwise Qty * UnitPrice.
    Robust to French headers and messy formats.
    """
    if df is None or df.empty:
        return df.assign(Montant_calculé=0.0), 0.0

    work = df.copy()

    # Try to find explicit total first
    col_total = _find_col(work, PRICE_ALIASES_TOTAL)
    col_unit  = _find_col(work, PRICE_ALIASES_UNIT)
    col_qty   = _find_col(work, QTY_ALIASES)

    line_amounts = []

    for _, row in work.iterrows():
        val = None
        if col_total:
            val = _to_float(row.get(col_total))
        if val is None and (col_unit and col_qty):
            q = _to_float(row.get(col_qty))
            pu = _to_float(row.get(col_unit))
            if q is not None and pu is not None:
                val = q * pu
        if val is None:
            val = 0.0
        line_amounts.append(val)

    work["Montant_calculé"] = line_amounts
    total = float(pd.Series(line_amounts).sum())
    return work, total


# ---------- Routes ----------
@app.route("/")
def index():
    # small dashboard: show last revisions and a few top elements
    latest_revs = Revision.query.order_by(Revision.uploaded_at.desc()).limit(10).all()
    total_elements = db.session.query(Element).count()
    return render_template("index.html", revisions=latest_revs, total_elements=total_elements)

@app.route("/upload-many", methods=["POST"])
def upload_many():
    try:
        files = request.files.getlist("files")
        forced_key = request.form.get("key")  # optional
        if not files:
            return jsonify({"error":"Please choose one or more .xlsx files."}), 400
        created = []
        for f in files:
            if not f.filename.lower().endswith(".xlsx"):
                return jsonify({"error": f"{f.filename}: only .xlsx allowed"}), 400
            rev = ingest_excel_to_revision(f, forced_key=forced_key)
            created.append({"id": rev.id, "name": rev.name})
        return jsonify({"ok": True, "created": created})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/elements")
def elements():
    q = db.session.execute(text("""
        SELECT
          e.key AS key,
          COUNT(s.id)        AS rev_count,
          GROUP_CONCAT(DISTINCT r.id)   AS rev_ids,
          GROUP_CONCAT(DISTINCT r.name) AS rev_names
        FROM elements e
        JOIN snapshots s ON s.element_id = e.id
        JOIN revisions r ON r.id = s.revision_id
        GROUP BY e.id, e.key
        ORDER BY e.key
        LIMIT 500
    """))
    rows = []
    for r in q.fetchall():
        rows.append({
            "key": r[0],
            "rev_count": r[1],
            "rev_ids": (r[2] or ""),
            "rev_names": (r[3] or "")
        })
    return render_template("elements.html", elements=rows)


@app.route("/element/<key>")
def element_view(key):
    el = Element.query.filter_by(key=key).first_or_404()
    snaps = (db.session.query(Snapshot, Revision)
             .join(Revision, Snapshot.revision_id==Revision.id)
             .filter(Snapshot.element_id==el.id)
             .order_by(Revision.uploaded_at.asc())
             .all())
    history = []
    last_data = None
    for snap, rev in snaps:
        data = json.loads(snap.data_json)
        history.append({"revision": rev.name, "uploaded_at": rev.uploaded_at, "data": data})
        last_data = data

    # build change log between consecutive snapshots
    changes = []
    for i in range(1, len(history)):
        before, after = history[i-1]["data"], history[i]["data"]
        diffs = diff_rows(before, after)
        if diffs:
            changes.append({
                "from": history[i-1]["revision"],
                "to": history[i]["revision"],
                "changes": diffs
            })

    # show a compact set of columns in table preview (union of keys minus _key)
    columns = sorted(set().union(*[set(h["data"].keys()) for h in history]) - {"_key"})
    return render_template("element.html", key=key, history=history, columns=columns, changes=changes)

@app.route("/revisions")
def revisions():
    revs = Revision.query.order_by(Revision.uploaded_at.desc()).all()
    return render_template("revisions.html", revisions=revs)

@app.post("/revisions/<int:rev_id>/delete")
def delete_revision(rev_id: int):
    # 1) Find revision or 404
    rev = Revision.query.get_or_404(rev_id)

    # 2) Delete all snapshots of that revision
    Snapshot.query.filter_by(revision_id=rev.id).delete(synchronize_session=False)

    # 3) Delete the revision itself
    db.session.delete(rev)
    db.session.commit()

    # 4) (Optional) prune elements that have no snapshots left
    db.session.execute(text("""
        DELETE FROM elements
        WHERE id NOT IN (SELECT DISTINCT element_id FROM snapshots)
    """))
    db.session.commit()

    return jsonify({"ok": True})
@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "GET":
        revs = Revision.query.order_by(Revision.uploaded_at.desc()).all()
        return render_template("compare.html", revisions=revs)

    # POST: compare baseline vs multiple targets
    try:
        rev1_id = int(request.form.get("rev1_id"))
        raw_ids = request.form.get("rev2_ids", "")
        rev2_ids = []
        for tok in raw_ids.split(","):
            tok = tok.strip()
            if tok:
                i = int(tok)
                if i not in rev2_ids and i != rev1_id:
                    rev2_ids.append(i)
        if not rev2_ids:
            raise ValueError("Pick at least one target revision.")

        df1 = df_from_revision(rev1_id)
        baseline = Revision.query.get(rev1_id)
        results = {}

        for rid in rev2_ids:
          df2 = df_from_revision(rid)
          added, deleted, modified = analyze_dataframes(df1, df2)
          rev = Revision.query.get(rid)
   
    # JSON-safe lists
          added_recs    = df_to_records(added)
          deleted_recs  = df_to_records(deleted)
          modified_recs = df_to_records(modified)

    # Clean + sort as requested
          added_recs    = sort_records_by_id(clean_added_deleted_records(added_recs))
          deleted_recs  = sort_records_by_id(clean_added_deleted_records(deleted_recs))
          modified_recs = sort_records_by_id(modified_recs)

          results[str(rid)] = {
            "meta": {
               "rev_id": rid,
               "rev_name": (rev.name if rev else f"#{rid}"),
               "rev_time": (rev.uploaded_at.isoformat() if rev else None),
               "added": len(added_recs),
               "deleted": len(deleted_recs),
               "modified": len(modified_recs),
            },
           "added":    added_recs,
           "deleted":  deleted_recs,
           "modified": modified_recs,
         }



        return jsonify({
            "ok": True,
            "baseline": {
                "rev_id": rev1_id,
                "rev_name": baseline.name if baseline else f"#{rev1_id}",
                "rev_time": baseline.uploaded_at.isoformat() if baseline else None
            },
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.post("/export_xlsx")
def export_xlsx():
    try:
        rev1_id = int(request.form["rev1_id"])
        rev2_id = int(request.form["rev2_id"])
        kind    = request.form.get("kind", "all").lower()

        df1 = df_from_revision(rev1_id)
        df2 = df_from_revision(rev2_id)
        added, deleted, modified = analyze_dataframes(df1, df2)

        # Prepare DataFrames per your rules
        def drop_dup_id(df):
            if df is None or df.empty:
                return df
            cols = ["N° de Prix"] + [c for c in df.columns if c != "N° de Prix" and norm_col(c) != "no de prix"]
            df = df.reindex(columns=[c for c in cols if c in df.columns])
            # numeric sort by N° de Prix
            def parse(x):
                try: return float(str(x).replace(",", "."))
                except: return float("inf")
            return df.sort_values(by="N° de Prix", key=lambda s: s.map(parse)) if "N° de Prix" in df.columns else df

        add_df = drop_dup_id(added)
        del_df = drop_dup_id(deleted)

        if modified is None or modified.empty:
            mod_df = modified
        else:
            order = [c for c in ["N° de Prix","Champ modifié","Ancienne valeur","Nouvelle valeur"] if c in modified.columns]
            mod_df = modified.reindex(columns=order)
            def parse(x):
                try: return float(str(x).replace(",", "."))
                except: return float("inf")
            if "N° de Prix" in mod_df.columns:
                mod_df = mod_df.sort_values(by=["N° de Prix","Champ modifié"],
                                            key=lambda s: s.map(parse) if s.name=="N° de Prix" else s)

        # Write to XLSX
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            if kind in ("added","deleted","modified"):
                {"added": add_df, "deleted": del_df, "modified": mod_df}[kind].to_excel(
                    writer, sheet_name=kind.title(), index=False
                )
            else:
                (add_df if add_df is not None else pd.DataFrame()).to_excel(writer, sheet_name="Added", index=False)
                (del_df if del_df is not None else pd.DataFrame()).to_excel(writer, sheet_name="Deleted", index=False)
                (mod_df if mod_df is not None else pd.DataFrame()).to_excel(writer, sheet_name="Modified", index=False)

        buf.seek(0)
        base = Revision.query.get(rev1_id)
        targ = Revision.query.get(rev2_id)
        base_name = base.name if base else f"{rev1_id}"
        targ_name = targ.name if targ else f"{rev2_id}"
        suffix = kind if kind in ("added","deleted","modified") else "all"
        filename = f"diff_{base_name}_vs_{targ_name}_{suffix}.xlsx"

        return send_file(buf,
            as_attachment=True,
            download_name=filename,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        app.logger.exception("export_xlsx failed")
        return jsonify({"error": str(e)}), 500
@app.get("/budget")
def budget_page():
    """Simple page to upload a bordereau/revision and check against a limit."""
    revs = Revision.query.order_by(Revision.uploaded_at.desc()).all()
    return render_template("budget.html", revisions=revs)

@app.post("/budget/upload")
def budget_upload_and_check():
    """
    Upload a single .xlsx, ingest as a new Revision,
    compute total, compare with provided limit, and return verdict.
    """
    try:
        f = request.files.get("file")
        limit_raw = request.form.get("limit", "").strip()
        forced_key = request.form.get("key") or None

        if not f or not f.filename.lower().endswith(".xlsx"):
            return jsonify({"error": "Please upload one .xlsx file."}), 400

        budget_limit = _to_float(limit_raw)
        if budget_limit is None:
            return jsonify({"error": "Please provide a numeric budget limit."}), 400

        # ingest -> new revision
        rev = ingest_excel_to_revision(f, forced_key=forced_key)

        # build df for this new revision
        df = df_from_revision(rev.id)

        # compute totals
        priced_df, total = compute_line_totals(df)

        # cache on revision (optional)
        rev.total_amount = total
        rev.over_budget = bool(total > budget_limit)
        db.session.commit()

        return jsonify({
            "ok": True,
            "revision_id": rev.id,
            "revision_name": rev.name,
            "total": round(total, 2),
            "limit": round(budget_limit, 2),
            "over_budget": total > budget_limit
        })
    except Exception as e:
        app.logger.exception("budget_upload_and_check failed")
        return jsonify({"error": str(e)}), 500

@app.get("/api/revision/<int:rev_id>/budget")
def budget_from_existing_rev(rev_id: int):
    """
    Compute budget totals using an already-ingested revision.
    Query param: ?limit=123456.78
    """
    try:
        rev = Revision.query.get_or_404(rev_id)
        limit_raw = request.args.get("limit", "").strip()
        budget_limit = _to_float(limit_raw) if limit_raw else None

        df = df_from_revision(rev.id)
        _, total = compute_line_totals(df)

        # cache (optional)
        if budget_limit is not None:
            rev.total_amount = total
            rev.over_budget = bool(total > budget_limit)
            db.session.commit()

        return jsonify({
            "ok": True,
            "revision_id": rev.id,
            "revision_name": rev.name,
            "total": round(total, 2),
            "limit": (round(budget_limit, 2) if budget_limit is not None else None),
            "over_budget": (total > budget_limit) if budget_limit is not None else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
# keep your imports, app, db, models, routes, etc. above …

def ensure_schema():
    """Create tables and add columns if missing (SQLite)."""
    with app.app_context():
        db.create_all()
        # use a transaction so ALTERs are committed
        with db.engine.begin() as con:
            cols = {row[1] for row in con.execute(text("PRAGMA table_info(revisions)"))}
            if "total_amount" not in cols:
                con.execute(text("ALTER TABLE revisions ADD COLUMN total_amount REAL"))
            if "over_budget" not in cols:
                con.execute(text("ALTER TABLE revisions ADD COLUMN over_budget INTEGER DEFAULT 0"))

    
@app.post("/pricebook/upload")
def pricebook_upload():
    try:
        f = request.files.get("file")
        forced = request.form.get("designation_key") or None
        if not f or not f.filename.lower().endswith(".xlsx"):
            return jsonify({"error":"Upload a .xlsx bordereau de prix."}), 400
        pb = ingest_pricebook(f, forced_designation_col=forced)
        return jsonify({"ok": True, "pricebook_id": pb.id, "pricebook_name": pb.name})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/revision/<int:rev_id>/budget_by_designation")
def api_rev_budget_by_designation(rev_id: int):
    try:
        limit_raw = request.args.get("limit", "").strip()
        budget_limit = _to_float(limit_raw) if limit_raw else None
        out = compute_revision_budget_by_designation(rev_id, budget_limit)
        if "error" in out:
            return jsonify(out), 400
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


# ---------- Init ----------
if __name__ == "__main__":
    ensure_schema()  # <-- now it's defined above
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)

