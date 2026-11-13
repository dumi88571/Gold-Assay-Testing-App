# Gold Assay Analyzer & ML Engine

A web app and analytical tool built for gold assay labs and mining sites. It predicts gold grades from raw spectrometer readings using a combination of physics and machine learning: a standard Beer-Lambert calibration curve handles the baseline, and a Gradient Boosting model corrects for interference from other minerals in the rock matrix. 

The app includes full lab batch processing, QA/QC tracking, calibration curve management, printable certificates, a REST API, and an MCP (Model Context Protocol) server for connecting to external tools.

---

## How It Works

Standard spectroscopy often runs into trouble when other elements in the rock (like high iron, copper, arsenic, or sulfur) distort the signal, or when the instrument drifts over a shift. This system breaks the calculation into two parts:

1. **Baseline Physics Calibration (Beer-Lambert Law):**
   * The app takes the ratio of the gold emission intensity against an internal standard ($I_{emission} / I_{standard}$).
   * It fits this ratio to known standard concentrations using a RANSAC regressor. This prevents bad calibration readings or dirty blanks from ruining the slope.
   * From the fit residuals, it calculates the analytical Limit of Detection (LOD) and Limit of Quantification (LOQ).

2. **Machine Learning Matrix Correction:**
   * Raw rock matrices cause signal suppression or enhancement. A Gradient Boosting model looks at the other elemental readings in the sample and calculates a correction factor.
   * Because grades near cutoff points need risk estimates, the model runs quantile regression ($\alpha = 0.1, 0.5, 0.9$). Instead of just returning a single number, it gives the lab tech an 80% confidence window for each sample.

---

## Machine Learning & Statistical Models

* **`sklearn.linear_model.RANSACRegressor`:** Used during calibration runs. It repeatedly samples subsets of standard readings to separate legitimate calibration points from contaminated outliers, producing an honest slope and intercept.
* **`sklearn.ensemble.GradientBoostingRegressor` (Quantile Loss):** Three gradient boosted tree ensembles trained on multi-element assay profiles. They predict the 10th percentile (conservative low estimate), the 50th percentile (median grade), and the 90th percentile (optimistic high estimate).
* **`shap` (TreeExplainer):** Breaks down the matrix adjustment so operators can see exactly which elements (e.g. high copper or arsenic) pushed the grade up or down compared to the baseline curve.
* **`sklearn.compose.ColumnTransformer` & `StandardScaler`:** Normalizes multi-element PPM values and spectral ratios into a clean scikit-learn pipeline before inference.

---

## Backend Architecture

The backend is built in Python using Flask's application factory pattern (`create_app`) and split into modular blueprints:

* **ML Engine (`app/services/ml_engine.py`):** Holds the `CalibrationModel` class, runs the inference pipeline, handles automated retraining when enough validated samples are logged, and computes uncertainty intervals.
* **Blueprints (`app/routes/`):**
  * `auth.py`: User registration, session handling, and role permissions (admin vs. lab tech) via Flask-Login.
  * `batch.py`: Handles batch sample uploads, parses tabular files with Pandas, and runs bulk inference.
  * `calibration.py`: Saves standard curves per instrument shift, plots slope/intercept lines, and monitors blank drift.
  * `analysis.py`: Single-sample lookup, manual assay entry, and historical rerun comparisons.
  * `statistics.py`: Calculates sample distribution stats, RSD percentages across duplicates, and shift summary reports.
  * `api.py`: JSON endpoints for external LIMS or automated lab balances.
* **Data Storage (`app/models/`):** SQLAlchemy ORM models covering `User`, `Sample`, `Batch`, `Calibration`, and audit trail logs. Runs on SQLite (`assay_professional.db`) out of the box for quick setup, and points to PostgreSQL in production via `DATABASE_URL`.
* **Data Validation (`app/schemas.py`):** Uses Pydantic models to catch malformed JSON payloads, missing columns, or impossible concentration values before they ever reach the database or ML engine.
* **MCP Server (`mcp_server.py`):** An asynchronous STDIO server using the official MCP Python SDK. It lets LLMs or automated agents query tools like `predict_gold_grade`, `batch_predict`, and `calibrate_model`.

---

## Frontend Architecture

The user interface uses clean, server-side rendered Jinja2 templates with vanilla CSS and responsive grid layouts. It is built to work smoothly on both standard lab workstations and tablets in field offices:

* **Dashboard (`templates/dashboard.html`):** Shows the current active calibration curve, today's sample count, average grade trends, and any batches flagged for QC review.
* **Batch Upload (`templates/batch_upload.html`, `batch_process.html`):** Drag-and-drop file uploader that takes CSV, Excel (`.xlsx`), or JSON. Lets the tech map columns before running predictions across the batch.
* **Sample View (`templates/sample_detail.html`):** Detailed breakdown of an individual sample. Displays the raw spectral intensity, the baseline Beer-Lambert calculation, the ML matrix adjustment, and the uncertainty bar chart.
* **Calibration Screen (`templates/calibration.html`):** Renders the standard series calibration curve with RANSAC inlier/outlier points marked, showing current slope, intercept, LOD, and LOQ values.
* **QC & Statistics (`templates/statistics.html`, `trends.html`):** Grade distribution histograms, duplicate variance checks, and correlation charts between gold and secondary elements.
* **Assay Certificate (`templates/certificate.html`):** A clean, print-ready HTML certificate for completed batches, formatted with batch metadata, operator sign-offs, and ISO 17025 style QC notes.

---

## Project Structure

```text
├── app/
│   ├── models/          # SQLAlchemy models (User, Sample, Batch, Calibration)
│   ├── routes/          # Flask blueprints (auth, batch, calibration, api, stats)
│   ├── services/        # ML engine and calibration math (ml_engine.py)
│   ├── templates/       # Jinja2 templates (dashboard, batch, certificates)
│   ├── schemas.py       # Pydantic validation schemas
│   └── utils/           # Helper functions for data parsing and reporting
├── mcp_server.py        # Model Context Protocol server for external tool calls
├── config.py            # App settings and QC threshold constants
├── run.py               # Main CLI runner and dev server entry point
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
└── vercel.json          # Vercel serverless configuration
```

---

## Setup & Local Run

### Prerequisites

* Python 3.10+
* Virtual environment (`venv`)

### 1. Clone & Install

```bash
git clone https://github.com/dumi88571/Gold-Assay-Testing-App.git
cd Gold-Assay-Testing-App

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory (optional for local testing; SQLite runs by default):

```ini
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=sqlite:///assay_professional.db
```

### 3. Initialize & Start

```bash
# Create database tables
python run.py --init-db

# Start the Flask app
python run.py
```

Open `http://127.0.0.1:5000` in your browser. The first account you register will automatically be given admin access.

---

## Useful CLI Commands

The `run.py` script has a couple of built-in flags for routine maintenance:

```bash
# Rebuild or initialize the database schema
python run.py --init-db

# Retrain the ML model once you have enough validated assay results
python run.py --retrain
```

---

## Running the MCP Server

If you want to use the assay engine as a tool inside an MCP client or AI workflow:

```bash
python mcp_server.py
```

This starts the server over standard input/output with three main tools:
* `predict_gold_grade`: Predicts grade and uncertainty bounds for a single sample.
* `batch_predict`: Runs predictions on a list of samples and checks for QC warnings.
* `calibrate_model`: Updates the active calibration slope and intercept.

---

## Deployment

* **Render:** Uses `render.yaml` with Gunicorn (`gunicorn run:app`).
* **Vercel:** Runs through `vercel.json` and the `index.py` serverless entry point.
* **On-Premise:** Can be run behind Nginx with Gunicorn and hooked up to a local lab PostgreSQL instance.

---

## License

MIT License.
