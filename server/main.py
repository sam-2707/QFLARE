from fastapi import FastAPI
from server.api.routes import router as api_router

app = FastAPI(title="QFLARE Central Server")
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "QFLARE server running"}
