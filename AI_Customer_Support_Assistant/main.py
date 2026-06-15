"""
main.py  —  Entry point
Run: uvicorn main:app --reload --port 8000
"""
import os
import sys

# Ensure project root is always on the path so all local packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from logger_config import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)

from api.app import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")