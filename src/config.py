import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory Paths
SAMPLES_DIR = PROJECT_ROOT / "samples"
RECORDS_DIR = PROJECT_ROOT / os.getenv("RECORDS_DIR", "records")
LEDGER_DIR = PROJECT_ROOT / os.getenv("LEDGER_DIR", "ledger_data")
LEDGER_FILE = LEDGER_DIR / "blockchain.json"

# Ensure runtime directories exist
RECORDS_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Vision API Credentials
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")

# If GOOGLE_APPLICATION_CREDENTIALS is set and relative, make it absolute
if GOOGLE_APPLICATION_CREDENTIALS and not Path(GOOGLE_APPLICATION_CREDENTIALS).is_absolute():
    GOOGLE_APPLICATION_CREDENTIALS = str(PROJECT_ROOT / GOOGLE_APPLICATION_CREDENTIALS)

# Face Matching Thresholds
# For 512-d normalized embeddings, cosine similarity >= 0.65 indicates strong facial match
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
