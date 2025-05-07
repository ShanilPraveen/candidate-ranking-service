from fastapi import FastAPI
from .routers import candidate

app = FastAPI()

app.include_router(candidate.router)
