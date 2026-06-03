from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.validate import router as validate_router

app = FastAPI(title="SOP Validation Agent")

app.include_router(upload_router)
app.include_router(validate_router)

@app.get("/")
def root():
    return {"message": "SOP Validation Agent Running"}