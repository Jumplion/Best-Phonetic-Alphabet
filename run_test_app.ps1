# -----------------------------
# Fix PowerShell execution policy (for current user only)
# -----------------------------
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass -Force

# -----------------------------
# Step 1: Create virtual environment if it doesn't exist
# -----------------------------
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}

# -----------------------------
# Step 2: Activate virtual environment
# -----------------------------
& .\.venv\Scripts\Activate.ps1

# -----------------------------
# Step 3: Install dependencies
# -----------------------------
python.exe -m pip install --upgrade pip
pip install -r requirements.txt

# -----------------------------
# Step 4: Create and populate the SQLite database
# (Adjust the file name if your DB setup script is named differently)
# -----------------------------
if (-Not (Test-Path "phoneme_data.db")) {
    python create_databases.py
}

# -----------------------------
# Step 5: Run the Flask application
# -----------------------------

# If using Flask 2.2+:
flask --app web-app/app.py --debug run

#   Flask < 2.2
# $env:FLASK_APP = "./web-app/app.py"
# $env:FLASK_ENV = "development"
# flask run
