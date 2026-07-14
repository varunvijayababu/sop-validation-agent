from fastapi import FastAPI

from app.api.upload import router as upload_router
from app.api.validate import router as validate_router

import logging
import os
import sys

# Set up logging handlers dynamically
handlers = [logging.StreamHandler()]

try:
    os.makedirs(
        "logs",
        exist_ok=True
    )
    handlers.append(logging.FileHandler("logs/app.log"))
except Exception as e:
    print(f"WARNING: Could not initialize file logging. Falling back to console-only logging. Error: {str(e)}", file=sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=handlers
)

logger = logging.getLogger(__name__)

logger.info(
    "Starting SOP Validation Agent"
)

app = FastAPI(
    title="SOP Validation Agent"
)

app.include_router(upload_router, tags=["Upload"])
app.include_router(validate_router, tags=["Validation"])

logger.info(
    "Routers loaded successfully"
)

@app.get("/")
def root():

    return {
        "message": "SOP Validation Agent Running"
    }