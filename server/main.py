from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from server.api.routes import router as api_router
from server.registry import registered_devices
import os

app = FastAPI(title="QFLARE Central Server")
app.include_router(api_router)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})

@app.post("/register", response_class=HTMLResponse)
def register_device(request: Request, device_id: str = Form(...)):
    if device_id in registered_devices:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Device already registered."})
    registered_devices.add(device_id)
    return RedirectResponse(url="/devices", status_code=303)

@app.get("/devices", response_class=HTMLResponse)
def list_devices(request: Request):
    return templates.TemplateResponse("devices.html", {"request": request, "devices": list(registered_devices)})
