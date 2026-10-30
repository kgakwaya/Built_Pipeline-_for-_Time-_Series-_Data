# Pipeline for Time-Series Data - Household Electricity Consumption

Formative 1 (Machine Learning Pipeline, Group 2). The group forecasts a household's **average active power one hour ahead**
(`Global_active_power`, kW) from the UCI *Individual household electric power consumption* data.

| Task | What | Where | Owner |
|---|---|---|---|
| 1 | EDA, preprocessing, lag / moving-average features, model + experiment table | `notebooks/`, `src/` | teammates |
| **2** | **Relational (MySQL) and MongoDB design, ERD, sample documents, queries + results** | `database/`, `docs/database_design.md` | Honette |
| 3 | CRUD + latest-record + date-range endpoints for both databases (FastAPI) | `api/` | Ketia |
| 4 | Prediction script: fetch from API → preprocess → load model → predict | `prediction/` | Ketia |

## Repository layout

```
database/
  sql/            01_schema.sql  02_seed_reference_data.sql  03_queries.sql
  erd/            erd.mmd (source)  erd.png  erd.svg
  mongo/          queries.py (8 queries)  sample_documents.json
  scripts/        load_sql.py  load_mongo.py  run_sql_queries.py  run_mongo_queries.py  export_sample_documents.py
  results/        sql_query_results.md  mongo_query_results.md      <- queries + their results (for the report)
  config.py       connection settings (env vars / .env)
data/processed/   hourly_observed.csv   (hourly data produced by Task 1)
docs/             database_design.md    (design rationale, table dictionary, results summary)
api/              main.py  sql_routes.py  mongo_routes.py  schemas.py  db.py          <- Task 3 (FastAPI)
prediction/       predict.py                                                       <- Task 4
tests/            test_pipeline.py (Task 1)  test_api.py (Tasks 3-4)
docker-compose.yml  requirements.txt  .env.example
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                     # edit passwords if you use your own servers
docker compose up -d                                     # MySQL 8 + MongoDB 7; or `podman compose up -d` (skip if you already run them)
```

## Task 2 - Databases (`database/`)

```bash
python -m database.scripts.load_sql                      # creates power_db, loads 34,589 hours + 136,604 circuit rows
python -m database.scripts.load_mongo                    # creates collections/indexes/validators, loads 34,589 hourly documents
python -m database.scripts.run_sql_queries               # -> database/results/sql_query_results.md
python -m database.scripts.run_mongo_queries             # -> database/results/mongo_query_results.md
```
Both loaders are re-runnable (they drop and recreate the data).

**Design**
* **MySQL** (6 tables): `households` → `circuits` / `hourly_readings` → `circuit_energy`, plus `ml_models` → `forecasts`. 3NF, FKs, CHECKs, unique (household, hour).
* **MongoDB** (5 collections): one document per hour with embedded `measurements`, `sub_metering_wh`, `quality`, `calendar`; pre-aggregated `daily_summaries`; `$jsonSchema` validators and unique compound indexes.
* Full rationale, table dictionary and a summary of all 17 queries (8 SQL, 9 MongoDB): [`docs/database_design.md`](docs/database_design.md).

## Task 3 - CRUD and time-series API (`api/`)

FastAPI service over the databases built in Task 2. Both backends expose **the same routes and the same flat JSON record**,
so a client can switch between `/sql` and `/mongo` without changing its code.

```bash
uvicorn api.main:app --reload         # after load_sql / load_mongo; open http://127.0.0.1:8000/docs (Swagger UI)
```

| Method | Route (`{db}` = `sql` or `mongo`) | Purpose |
|---|---|---|
| POST | `/{db}/readings` | create an hourly reading (201; **409** if that hour exists; 422 on invalid values) |
| GET | `/{db}/readings?limit=&offset=` | list readings in time order |
| GET | `/{db}/readings/latest?valid_only=true` | **latest record** (`valid_only` skips incomplete hours such as the final one) |
| GET | `/{db}/readings/range?start=&end=&valid_only=` | **records by date range**, both ends inclusive, oldest first |
| GET | `/{db}/readings/{id}` | one reading (SQL: integer `reading_id`; Mongo: `ObjectId` string) |
| PUT | `/{db}/readings/{id}` | update the fields that are sent |
| DELETE | `/{db}/readings/{id}` | delete (SQL cascades to `circuit_energy`) |
| POST / GET | `/{db}/forecasts` | store / list predictions (used by Task 4) |
| GET | `/health` | reports whether MySQL and MongoDB are reachable |

Design notes
* **SQL**: a reading spans `hourly_readings` + `circuit_energy` (long format). Writes touch both inside one transaction;
  reads pivot the circuits back into columns. **Mongo**: the Task 2 nested layout (`measurements`, `sub_metering_wh`, `quality`, `calendar`).
* Derived fields follow the Task 1 rules: `is_valid_hour = n_obs_minutes >= 45`, and
  `other_energy_wh = global_active_power_kw*1000 - (sub_metering_1+2+3)` (recomputed on create/update unless sent explicitly).
* Validation: timestamps must be on the hour and time-zone free, values non-negative; duplicate hours → 409; database down → 503.
* Note: the pre-aggregated Mongo `daily_summaries` are not recomputed by the CRUD routes.

Example requests:
```bash
curl "http://127.0.0.1:8000/sql/readings/latest?valid_only=true"
curl "http://127.0.0.1:8000/mongo/readings/range?start=2010-11-26T00:00:00&end=2010-11-26T05:00:00"
curl -X POST http://127.0.0.1:8000/mongo/readings -H "Content-Type: application/json" \
     -d '{"timestamp":"2010-11-27T00:00:00","global_active_power_kw":1.25,"voltage_v":240.3,"global_intensity_a":5.2,"global_reactive_power_kvar":0.11,"sub_metering_1_wh":0,"sub_metering_2_wh":30,"sub_metering_3_wh":600}'
curl -X PUT http://127.0.0.1:8000/mongo/readings/<id> -H "Content-Type: application/json" -d '{"global_active_power_kw":1.4}'
curl -X DELETE http://127.0.0.1:8000/mongo/readings/<id>
```

## Task 4 - Prediction script (`prediction/predict.py`)

```bash
python -m prediction.predict                          # MySQL backend, origin = latest valid hour
python -m prediction.predict --backend mongo --save   # MongoDB, and store the forecast in `forecasts`
python -m prediction.predict --origin 2010-06-15T18:00:00 --show-features   # any historical hour; prints the 31 inputs
```

1. **Fetch** - `GET /{db}/readings/latest?valid_only=true` gives the forecast origin; `GET /{db}/readings/range`
   fetches the preceding 195 h (longest feature look-back 168 h + fill limit + margin).
2. **Preprocess** - reuses `src/` from Task 1: regular hourly grid (absent hours → NaN), 45-minute quality rule,
   `Other_energy_Wh` re-derived, causal forward fill (≤ 3 h), then the same lag (1-168 h), rolling mean/std (3-168 h),
   exogenous lag-1 and calendar features, in the exact column order saved in `best_model_metadata.json`.
   If a gap longer than 3 h leaves a feature undefined, the script stops with a clear message instead of guessing.
3. **Load** `outputs/models/best_model.joblib` (E4, tuned Random Forest).
4. **Predict** `Global_active_power` for origin + 1 h; prints it with the persistence baseline and, because the data are
   historical, the stored actual value and error when available. `--save` POSTs it to `/{db}/forecasts` (re-running replaces it).

The script reproduces the Task 1 notebook's test-set predictions (max difference 5e-5 kW over 300 sampled hours, i.e. rounding).

## Tests

```bash
pytest tests -v
```
`tests/test_pipeline.py` covers Task 1. `tests/test_api.py` runs the Mongo CRUD routes and an **end-to-end Task 3 → Task 4**
test (HTTP fetch → preprocessing → model → stored forecast) against an in-memory MongoDB (`mongomock`), so no server is needed;
the MySQL route tests run automatically when `power_db` is reachable and are skipped otherwise.
